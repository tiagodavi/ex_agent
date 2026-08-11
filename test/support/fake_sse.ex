defmodule ExAgent.Test.FakeSSE do
  @moduledoc """
  A fake `Req` adapter that delivers an SSE body as *separate* `{:data, _}`
  messages.

  Req's plug adapter emits the whole body as one chunk
  (`deps/req/lib/req/steps.ex`: "the entire response body is emitted as one
  chunk"), so a plug stub physically cannot exercise the split-frame
  reassembly path in `ExAgent.Services.Streaming`. This adapter can.

  It drives the real receive loop and the real `Req.parse_message/2` path — only
  the socket is replaced.

      req = Req.new(adapter: FakeSSE.adapter(["data: {\\"a\\":1}\\n", "\\n"]))
  """

  @doc """
  Builds an adapter that sends each element of `parts` as its own data message.

  ## Options

  - `:status` - response status (default `200`)
  - `:error` - a transport reason delivered after `parts`, instead of a clean
    close, to simulate a mid-stream failure
  """
  @spec adapter([binary()], keyword()) ::
          (Req.Request.t() -> {Req.Request.t(), Req.Response.t()})
  def adapter(parts, opts \\ []) do
    status = Keyword.get(opts, :status, 200)
    error = Keyword.get(opts, :error)

    fn request ->
      ref = make_ref()
      owner = self()

      Enum.each(parts, fn part -> send(owner, {__MODULE__, ref, {:data, part}}) end)

      if error do
        send(owner, {__MODULE__, ref, {:error, error}})
      else
        send(owner, {__MODULE__, ref, :done})
      end

      response = %Req.Response{
        status: status,
        headers: %{"content-type" => ["text/event-stream"]},
        body: %Req.Response.Async{
          pid: owner,
          ref: ref,
          stream_fun: &stream_fun/2,
          cancel_fun: &cancel_fun/1
        }
      }

      {request, response}
    end
  end

  @doc """
  Splits a binary into `count` roughly equal parts.
  """
  @spec split(binary(), pos_integer()) :: [binary()]
  def split(binary, count) do
    size = max(div(byte_size(binary), count), 1)

    binary
    |> :binary.bin_to_list()
    |> Enum.chunk_every(size)
    |> Enum.map(&:binary.list_to_bin/1)
  end

  @doc """
  Every possible two-way split of a binary — the exhaustive boundary case.
  """
  @spec all_splits(binary()) :: [[binary()]]
  def all_splits(binary) do
    for at <- 1..(byte_size(binary) - 1)//1 do
      <<first::binary-size(at), second::binary>> = binary
      [first, second]
    end
  end

  # Matches the shape Req expects of a stream_fun: decode one mailbox message.
  defp stream_fun(ref, {__MODULE__, ref, {:data, data}}), do: {:ok, [data: data]}
  defp stream_fun(ref, {__MODULE__, ref, :done}), do: {:ok, [:done]}
  defp stream_fun(ref, {__MODULE__, ref, {:error, reason}}), do: {:error, reason}
  defp stream_fun(_ref, _message), do: :unknown

  defp cancel_fun(_ref), do: :ok
end
