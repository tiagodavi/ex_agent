defmodule ExAgent.Chunk do
  @moduledoc """
  One event in a streamed response.

  `ExAgent.chat_stream/3` and `ExAgent.Provider.stream/3` yield these instead of
  bare strings, so usage, finish reasons, reasoning traces, and tool-call deltas
  are all available rather than discarded.

      agent
      |> ExAgent.chat_stream("Explain OTP")
      |> Enum.each(fn
        %ExAgent.Chunk{type: :text_delta, text: text} -> IO.write(text)
        %ExAgent.Chunk{type: :done, finish_reason: reason} -> IO.puts("\\n[\#{reason}]")
        _chunk -> :ok
      end)

  Use `ExAgent.collect/1` to fold a chunk stream back into the same
  `ExAgent.Response` that `ExAgent.chat/3` returns.

  ## Types

  | Type | Carries |
  |------|---------|
  | `:text_delta` | `:text` - a piece of the assistant's answer |
  | `:thinking_delta` | `:text` - a piece of the model's reasoning trace |
  | `:tool_call_delta` | `:index`, `:id`, `:name`, `:arguments` |
  | `:usage` | `:usage` - token counts |
  | `:done` | `:finish_reason`, and `:error` when the stream failed |

  Every stream ends with exactly one `:done` chunk, including when it fails.
  That invariant is what lets `ExAgent.collect/1` always return a result.

  ## Tool-call arguments arrive in pieces

  `:arguments` is a **raw JSON fragment**, not a decoded map. Providers emit
  function arguments split across many chunks, to be concatenated by `:index`
  before decoding. `:name` arrives once, on the first fragment of a call.
  `ExAgent.collect/1` does this reassembly for you.
  """

  alias ExAgent.Error

  @type type :: :text_delta | :thinking_delta | :tool_call_delta | :usage | :done

  @type finish_reason :: :stop | :length | :tool_calls | :content_filter | :error | nil

  @type t :: %__MODULE__{
          type: type(),
          text: String.t() | nil,
          index: non_neg_integer() | nil,
          id: String.t() | nil,
          name: String.t() | nil,
          arguments: String.t() | nil,
          usage: map() | nil,
          finish_reason: finish_reason(),
          error: Error.t() | nil
        }

  defstruct [:type, :text, :index, :id, :name, :arguments, :usage, :finish_reason, :error]

  @doc "Builds a text delta chunk."
  @spec text_delta(String.t()) :: t()
  def text_delta(text), do: %__MODULE__{type: :text_delta, text: text}

  @doc "Builds a reasoning-trace delta chunk."
  @spec thinking_delta(String.t()) :: t()
  def thinking_delta(text), do: %__MODULE__{type: :thinking_delta, text: text}

  @doc "Builds a usage chunk."
  @spec usage(map()) :: t()
  def usage(usage), do: %__MODULE__{type: :usage, usage: usage}

  @doc "Builds the terminal chunk."
  @spec done(finish_reason(), Error.t() | nil) :: t()
  def done(finish_reason \\ :stop, error \\ nil),
    do: %__MODULE__{type: :done, finish_reason: finish_reason, error: error}

  @doc """
  Builds a terminal chunk carrying an error.

  Streams never raise mid-flight: whatever was already emitted stays valid, and
  the failure arrives as the final chunk.
  """
  @spec error(Error.t()) :: t()
  def error(%Error{} = error), do: done(:error, error)

  @doc """
  Reassembles `:tool_call_delta` chunks into completed tool calls.

  Fragments are grouped by `:index`, concatenated in arrival order, and decoded.
  A fragment set that does not parse as JSON is kept as `%{"raw" => fragment}`
  rather than discarded, matching how non-streaming responses handle the same
  case.

  Returns `nil` when there were no tool calls, so it can be handed straight to
  `ExAgent.Message.new/1`.

  ## Examples

      iex> chunks = [
      ...>   %ExAgent.Chunk{type: :tool_call_delta, index: 0, name: "sum", arguments: ~s({"a":)},
      ...>   %ExAgent.Chunk{type: :tool_call_delta, index: 0, arguments: ~s(1})}
      ...> ]
      iex> ExAgent.Chunk.finalize_tool_calls(chunks)
      [%{"name" => "sum", "args" => %{"a" => 1}}]
  """
  @spec finalize_tool_calls([t()]) :: [map()] | nil
  def finalize_tool_calls([]), do: nil

  def finalize_tool_calls(chunks) do
    chunks
    |> Enum.filter(&(&1.type == :tool_call_delta))
    |> Enum.group_by(& &1.index)
    |> Enum.sort_by(fn {index, _fragments} -> index end)
    |> Enum.map(fn {_index, fragments} -> build_tool_call(fragments) end)
    |> case do
      [] -> nil
      calls -> calls
    end
  end

  @spec build_tool_call([t()]) :: map()
  defp build_tool_call(fragments) do
    name = Enum.find_value(fragments, & &1.name)

    arguments =
      fragments
      |> Enum.map(&(&1.arguments || ""))
      |> IO.iodata_to_binary()

    call = %{"name" => name, "args" => decode_arguments(arguments)}

    # Only when the provider issued one, so a call correlates back by id where
    # that matters and the shape stays minimal where it does not.
    case Enum.find_value(fragments, & &1.id) do
      nil -> call
      id -> Map.put(call, "id", id)
    end
  end

  @spec decode_arguments(String.t()) :: map()
  defp decode_arguments(""), do: %{}

  defp decode_arguments(arguments) do
    case Jason.decode(arguments) do
      {:ok, decoded} when is_map(decoded) -> decoded
      _other -> %{"raw" => arguments}
    end
  end

  @doc """
  Normalizes a provider's finish-reason string.

  Unrecognized values become `:stop` rather than being dropped, since the stream
  did in fact end.

  ## Examples

      iex> ExAgent.Chunk.finish_reason("tool_calls")
      :tool_calls

      iex> ExAgent.Chunk.finish_reason("MAX_TOKENS")
      :length
  """
  @spec finish_reason(String.t() | nil) :: finish_reason()
  def finish_reason(nil), do: nil

  def finish_reason(reason) when is_binary(reason) do
    case String.downcase(reason) do
      "stop" -> :stop
      "end_turn" -> :stop
      "length" -> :length
      "max_tokens" -> :length
      "tool_calls" -> :tool_calls
      "function_call" -> :tool_calls
      "content_filter" -> :content_filter
      "safety" -> :content_filter
      _other -> :stop
    end
  end
end
