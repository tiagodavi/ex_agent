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
