defprotocol ExAgent.LlmProvider do
  @moduledoc """
  Protocol for standardizing LLM provider interactions.

  Any LLM provider can be integrated by defining a struct and
  implementing this protocol. The protocol handles chat completions
  with optional file attachments via the `ExAgent.Message` struct.

  ## Implementations

  The library provides built-in implementations for:
  - `ExAgent.Providers.OpenAI`
  - `ExAgent.Providers.Gemini`
  - `ExAgent.Providers.DeepSeek`

  ## Extensibility

  To add a new provider, define a struct and implement this protocol:

      defmodule MyApp.Providers.CustomLLM do
        defstruct [:api_key, :model, :base_url, :system_prompt, :req, tools: []]

        defimpl ExAgent.LlmProvider do
          def chat(provider, messages, opts) do
            # Your implementation here
          end
        end
      end
  """

  @doc """
  Sends a list of messages to the LLM and returns the assistant's response.

  Messages may include file attachments via the `attachments` field.
  Returns `{:ok, message}` for a regular response, `{:tool_call, name, args}`
  when the LLM wants to invoke a tool, or `{:error, reason}` on failure.
  """
  @spec chat(t(), [ExAgent.Message.t()], keyword()) ::
          {:ok, ExAgent.Message.t()}
          | {:tool_call, String.t(), map()}
          | {:error, term()}
  def chat(provider, messages, opts \\ [])
end
