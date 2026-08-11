defmodule ExAgent.Test.MinimalProvider do
  @moduledoc """
  A provider implementing only the required `chat/3` callback.

  Exists so the dispatcher's fallbacks stay covered - the `[:text]` modality
  default and the `:unsupported` errors for `upload/4`, `stream/3` and `embed/3`
  - without depending on a shipped provider that happens to omit them today.
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
end
