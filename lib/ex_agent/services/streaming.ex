defmodule ExAgent.Services.Streaming do
  @moduledoc """
  Shared SSE stream transport for LLM providers.

  Wraps a `Req` request made with `into: :self` in a lazy `Stream` of
  `ExAgent.Chunk` structs. Framing lives in `ExAgent.SSE`; this module owns the
  transport, so provider services only supply a mapper turning one decoded frame
  into zero or more chunks.

  Nothing here raises. A non-200 response, a transport failure, or an idle
  timeout all terminate the stream with a `:done` chunk carrying an
  `ExAgent.Error` — whatever was already emitted stays valid.
  """

  alias ExAgent.{Chunk, Error, SSE}

  # One frame can legitimately carry text, usage, and a finish reason at once,
  # so a mapper returns a list rather than a single value.
  @type mapper :: (map() -> [Chunk.t()])

  # Long enough for a slow model to think, short enough that a dead connection
  # does not hang a caller forever.
  @idle_timeout :timer.minutes(5)

  @doc """
  Streams an SSE response as a lazy enumerable of `ExAgent.Chunk` structs.

  `req_opts` are passed to `Req.post/2`; `into: :self` is added automatically.
  `provider` names the module in any error chunk. The stream always ends with
  exactly one `:done` chunk.
  """
  @spec stream(Req.Request.t(), keyword(), module(), mapper()) :: Enumerable.t()
  def stream(req, req_opts, provider, mapper) do
    Stream.resource(
      fn -> start(req, req_opts, provider) end,
      fn state -> next(state, mapper) end,
      &stop/1
    )
  end

  @spec start(Req.Request.t(), keyword(), module()) :: map() | {:fail, Chunk.t()}
  defp start(req, req_opts, provider) do
    case Req.post(req, [into: :self] ++ req_opts) do
      {:ok, %Req.Response{status: 200} = resp} ->
        %{resp: resp, buffer: "", finish_reason: nil, provider: provider}

      {:ok, %Req.Response{status: status} = resp} ->
        body = drain(resp, "")
        safe_cancel(resp)
        {:fail, Chunk.error(Error.from_http(status, decode_body(body), provider))}

      {:error, reason} ->
        {:fail, Chunk.error(Error.from_transport(reason, provider))}
    end
  end

  # A failure discovered before any data arrived still emits one terminal chunk.
  defp next({:fail, chunk}, _mapper), do: {[chunk], :halted}
  defp next(:halted, _mapper), do: {:halt, :halted}

  defp next(%{resp: resp, buffer: buffer} = state, mapper) do
    receive do
      message ->
        case Req.parse_message(resp, message) do
          {:ok, messages} ->
            {data, transport_done?} = reduce_messages(messages)
            {frames, rest} = SSE.decode(buffer <> data)
            {mapped, sse_done?} = map_frames(frames, mapper)

            state = %{state | buffer: rest, finish_reason: last_finish(mapped, state)}

            # A mapper turns a finish-reason frame into its own `:done`, but the
            # single terminal chunk is this module's to emit — the reason has
            # already been captured above. Keeping both would end every real
            # stream with two `:done` chunks.
            chunks = Enum.reject(mapped, &(&1.type == :done))

            cond do
              transport_done? or sse_done? -> {chunks ++ [terminal(state)], :halted}
              chunks != [] -> {chunks, state}
              true -> next(state, mapper)
            end

          {:error, reason} ->
            {[Chunk.error(Error.from_transport(reason, state.provider))], :halted}

          :unknown ->
            next(state, mapper)
        end
    after
      @idle_timeout ->
        error = Error.new(:timeout, "stream idle for 5 minutes with no data", state.provider)
        {[Chunk.error(error)], :halted}
    end
  end

  @spec terminal(map()) :: Chunk.t()
  defp terminal(%{finish_reason: reason}), do: Chunk.done(reason || :stop)

  # Remember a finish reason seen mid-stream so the terminal chunk can carry it.
  @spec last_finish([Chunk.t()], map()) :: Chunk.finish_reason()
  defp last_finish(chunks, state) do
    Enum.reduce(chunks, state.finish_reason, fn
      %Chunk{finish_reason: reason}, _acc when not is_nil(reason) -> reason
      _chunk, acc -> acc
    end)
  end

  @spec map_frames([SSE.frame()], mapper()) :: {[Chunk.t()], boolean()}
  defp map_frames(frames, mapper) do
    Enum.reduce(frames, {[], false}, fn
      :done, {chunks, _done?} ->
        {chunks, true}

      frame, {chunks, done?} ->
        mapped = frame |> mapper.() |> Enum.reject(&is_nil/1)
        {chunks ++ mapped, done?}
    end)
  end

  @spec reduce_messages([term()]) :: {binary(), boolean()}
  defp reduce_messages(messages) do
    Enum.reduce(messages, {"", false}, fn
      {:data, data}, {acc, done?} -> {acc <> data, done?}
      :done, {acc, _done?} -> {acc, true}
      _other, acc -> acc
    end)
  end

  @spec decode_body(binary()) :: term()
  defp decode_body(body) do
    case Jason.decode(body) do
      {:ok, decoded} -> decoded
      {:error, _reason} -> body
    end
  end

  defp drain(resp, acc) do
    receive do
      message ->
        case Req.parse_message(resp, message) do
          {:ok, messages} ->
            {data, done?} = reduce_messages(messages)
            acc = acc <> data
            if done?, do: acc, else: drain(resp, acc)

          _other ->
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
