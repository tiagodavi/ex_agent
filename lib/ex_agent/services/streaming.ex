defmodule ExAgent.StreamError do
  @moduledoc """
  Raised when a streaming request fails (non-200 status or transport error).

  The error surfaces lazily, when the stream is consumed.
  """

  defexception [:status, :body]

  @impl true
  def message(%{status: nil, body: body}) do
    "stream request failed: #{inspect(body)}"
  end

  def message(%{status: status, body: body}) do
    "stream request failed with status #{status}: #{inspect(body)}"
  end
end

defmodule ExAgent.Services.Streaming do
  @moduledoc """
  Shared Server-Sent Events (SSE) streaming helper for LLM providers.

  Wraps a `Req` request made with `into: :self` in a lazy `Stream` that yields
  text chunks. SSE framing (buffering split frames, `data:` prefixes, `[DONE]`
  termination) lives here so provider services only supply a per-provider parser
  that extracts the text delta from a decoded frame.
  """

  alias ExAgent.StreamError

  @type chunk_parser :: (map() -> String.t() | nil)

  @doc """
  Streams an SSE response as a lazy enumerable of text chunks.

  `req_opts` are passed to `Req.post/2` (e.g. `url:`, `json:`, `receive_timeout:`);
  `into: :self` is added automatically. `chunk_parser` receives each decoded SSE
  frame (a map) and returns the text to emit, or `nil` to skip it.

  Raises `ExAgent.StreamError` (when consumed) on a non-200 response.
  """
  @spec stream(Req.Request.t(), keyword(), chunk_parser()) :: Enumerable.t()
  def stream(req, req_opts, chunk_parser) do
    Stream.resource(
      fn -> start(req, req_opts) end,
      fn state -> next(state, chunk_parser) end,
      &stop/1
    )
  end

  @doc false
  # Splits an SSE buffer into complete events plus the trailing partial event.
  # Events are separated by a blank line ("\n\n"); the last element is always the
  # (possibly empty) incomplete remainder held back for the next chunk.
  @spec take_events(binary()) :: {[binary()], binary()}
  def take_events(buffer) do
    parts = String.split(buffer, "\n\n")
    {complete, [rest]} = Enum.split(parts, -1)
    {complete, rest}
  end

  @doc false
  # Extracts and joins the payload of all `data:` lines in one SSE event.
  # Returns `:done` for a `[DONE]` sentinel, `nil` when there is no data.
  @spec event_data(binary()) :: binary() | :done | nil
  def event_data(event) do
    data =
      event
      |> String.split("\n")
      |> Enum.filter(&String.starts_with?(&1, "data:"))
      |> Enum.map_join("", fn line ->
        line |> String.replace_prefix("data:", "") |> String.trim_leading()
      end)

    case data do
      "" -> nil
      "[DONE]" -> :done
      json -> json
    end
  end

  defp start(req, req_opts) do
    resp = Req.post!(req, [into: :self] ++ req_opts)

    if resp.status == 200 do
      %{resp: resp, buffer: "", done: false}
    else
      body = drain(resp, "")
      safe_cancel(resp)
      raise StreamError, status: resp.status, body: body
    end
  end

  defp next(%{done: true} = state, _parser), do: {:halt, state}

  defp next(%{resp: resp, buffer: buffer} = state, parser) do
    receive do
      message ->
        case Req.parse_message(resp, message) do
          {:ok, chunks} ->
            {data, done?} = reduce_chunks(chunks)
            {events, rest} = take_events(buffer <> data)
            {texts, sse_done?} = parse_events(events, parser)
            state = %{state | buffer: rest, done: done? or sse_done?}

            cond do
              texts != [] -> {texts, state}
              state.done -> {:halt, state}
              true -> next(state, parser)
            end

          {:error, reason} ->
            raise StreamError, status: nil, body: reason

          :unknown ->
            next(state, parser)
        end
    after
      :timer.minutes(5) -> {:halt, state}
    end
  end

  defp parse_events(events, parser) do
    Enum.reduce(events, {[], false}, fn event, {texts, done?} ->
      case event_data(event) do
        :done ->
          {texts, true}

        nil ->
          {texts, done?}

        json ->
          case Jason.decode(json) do
            {:ok, decoded} ->
              case parser.(decoded) do
                text when is_binary(text) and text != "" -> {texts ++ [text], done?}
                _ -> {texts, done?}
              end

            {:error, _} ->
              {texts, done?}
          end
      end
    end)
  end

  defp reduce_chunks(chunks) do
    Enum.reduce(chunks, {"", false}, fn
      {:data, data}, {acc, done?} -> {acc <> data, done?}
      :done, {acc, _done?} -> {acc, true}
      _other, acc -> acc
    end)
  end

  defp drain(resp, acc) do
    receive do
      message ->
        case Req.parse_message(resp, message) do
          {:ok, chunks} ->
            {data, done?} = reduce_chunks(chunks)
            acc = acc <> data
            if done?, do: acc, else: drain(resp, acc)

          _ ->
            acc
        end
    after
      5_000 -> acc
    end
  end

  defp stop(%{resp: resp}), do: safe_cancel(resp)
  defp stop(_state), do: :ok

  defp safe_cancel(resp) do
    Req.cancel_async_response(resp)
    :ok
  rescue
    _ -> :ok
  catch
    _, _ -> :ok
  end
end
