defmodule ExAgent.Context do
  @moduledoc """
  Portable conversation state shared across agents and patterns.

  Holds the message history, metadata, and an optional parent reference
  for linking back to an orchestrator in the subagent pattern.
  """

  alias ExAgent.Message

  @type t :: %__MODULE__{
          messages: [Message.t()],
          metadata: map(),
          parent_ref: reference() | nil
        }

  defstruct messages: [], metadata: %{}, parent_ref: nil

  @doc """
  Creates a new context with optional initial values.

  ## Examples

      iex> ctx = ExAgent.Context.new()
      iex> ctx.messages
      []

      iex> ctx = ExAgent.Context.new(metadata: %{session: "abc"})
      iex> ctx.metadata
      %{session: "abc"}
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{
      messages: opts[:messages] || [],
      metadata: opts[:metadata] || %{},
      parent_ref: opts[:parent_ref]
    }
  end

  @doc """
  Appends a message to the context.

  ## Examples

      iex> {:ok, msg} = ExAgent.Message.new(role: :user, content: "Hello")
      iex> ctx = ExAgent.Context.new() |> ExAgent.Context.add_message(msg)
      iex> length(ctx.messages)
      1
  """
  @spec add_message(t(), Message.t()) :: t()
  def add_message(%__MODULE__{messages: messages} = context, %Message{} = message) do
    %{context | messages: messages ++ [message]}
  end

  @doc """
  Drops the oldest messages, keeping at most `max` of them.

  Conversation history otherwise grows without bound: every turn resends the
  whole transcript, so cost climbs turn over turn until the model returns
  `:context_length` and the agent is stuck. Trimming is opt-in - silently
  forgetting what a user said is a decision the caller has to make.

  Leading `:system` messages are always kept: they carry instructions that must
  outlive the window. A `:tool` result is never left without the `:assistant`
  message that requested it, since providers reject an orphaned result.

  ## Examples

      iex> {:ok, a} = ExAgent.Message.new(role: :user, content: "one")
      iex> {:ok, b} = ExAgent.Message.new(role: :user, content: "two")
      iex> ctx = ExAgent.Context.new(messages: [a, b])
      iex> ExAgent.Context.trim(ctx, 1).messages |> Enum.map(& &1.content)
      ["two"]
  """
  @spec trim(t(), pos_integer()) :: t()
  def trim(%__MODULE__{messages: messages} = context, max) when is_integer(max) and max > 0 do
    {system, rest} = Enum.split_while(messages, &(&1.role == :system))

    kept =
      rest
      |> Enum.take(-max)
      |> drop_orphaned_results()

    %{context | messages: system ++ kept}
  end

  # A `:tool` message at the head lost the assistant turn that called it.
  @spec drop_orphaned_results([Message.t()]) :: [Message.t()]
  defp drop_orphaned_results([%Message{role: :tool} | rest]), do: drop_orphaned_results(rest)
  defp drop_orphaned_results(messages), do: messages

  @doc """
  Returns the last assistant message from the context, or nil if none exists.

  ## Examples

      iex> ExAgent.Context.get_last_assistant_message(ExAgent.Context.new())
      nil
  """
  @spec get_last_assistant_message(t()) :: Message.t() | nil
  def get_last_assistant_message(%__MODULE__{messages: messages}) do
    messages
    |> Enum.reverse()
    |> Enum.find(&(&1.role == :assistant))
  end
end
