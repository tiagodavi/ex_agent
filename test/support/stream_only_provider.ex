defmodule ExAgent.Test.StreamOnlyProvider do
  @moduledoc """
  A provider that streams but has not opted into structured output.

  Exists so the `:schema` gate on `ExAgent.Provider.stream/3` stays covered.
  `ExAgent.Test.MinimalProvider` cannot serve here: it has no `stream/3` at all,
  so it fails on the capability check before any schema is considered.
  """

  @behaviour ExAgent.Provider

  defstruct [:api_key]

  @type t :: %__MODULE__{api_key: String.t() | nil}

  @spec new(keyword()) :: t()
  def new(opts \\ []), do: struct!(__MODULE__, opts)

  @impl true
  def chat(_provider, _messages, _opts \\ []) do
    {:ok, message} = ExAgent.Message.new(role: :assistant, content: "ok")
    {:ok, ExAgent.Response.new(message)}
  end

  @impl true
  def stream(_provider, _messages, _opts \\ []), do: [ExAgent.Chunk.done(:stop)]
end
