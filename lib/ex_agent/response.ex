defmodule ExAgent.Response do
  @moduledoc """
  A completed assistant turn.

  Returned by `ExAgent.chat/3` and by `ExAgent.collect/1` on a chunk stream, so
  streaming and non-streaming share one downstream code path.

      {:ok, response} = ExAgent.chat(agent, "Explain OTP")
      response.content            #=> "OTP is ..."
      response.usage.total_tokens #=> 412
      response.finish_reason      #=> :stop

  `:message` holds the `ExAgent.Message` appended to conversation history;
  `:content` is a shortcut for `response.message.content`.

  `:thinking` carries a reasoning trace when the model emitted one. It is
  deliberately kept out of `:message` - reasoning is not conversation history,
  and replaying it corrupts the next turn.
  """

  alias ExAgent.Message

  @type usage :: %{
          optional(:input_tokens) => non_neg_integer() | nil,
          optional(:output_tokens) => non_neg_integer() | nil,
          optional(:total_tokens) => non_neg_integer() | nil
        }

  @type t :: %__MODULE__{
          content: String.t(),
          message: Message.t(),
          tool_calls: [map()] | nil,
          thinking: String.t() | nil,
          usage: usage(),
          finish_reason: ExAgent.Chunk.finish_reason()
        }

  @enforce_keys [:content, :message]
  defstruct [:content, :message, :tool_calls, :thinking, :finish_reason, usage: %{}]

  @doc """
  Wraps an assistant message in a response.

  ## Examples

      iex> {:ok, message} = ExAgent.Message.new(role: :assistant, content: "hi")
      iex> ExAgent.Response.new(message).content
      "hi"
  """
  @spec new(Message.t(), keyword()) :: t()
  def new(%Message{} = message, attrs \\ []) do
    %__MODULE__{
      content: message.content,
      message: message,
      tool_calls: message.tool_calls,
      thinking: attrs[:thinking],
      usage: attrs[:usage] || %{},
      finish_reason: attrs[:finish_reason]
    }
  end
end
