defmodule ExAgent.Provider do
  @moduledoc """
  Behaviour for LLM providers.

  A provider is an interchangeable *service* (OpenAI, Gemini, DeepSeek, ...)
  implementing a shared capability. Each provider is a struct module that
  declares `@behaviour ExAgent.Provider` and implements `chat/3` (required)
  and optionally `upload/4`. The provider struct carries its own config
  (api key, model, cached Req client) and is passed as the first argument.

  This module also acts as the dispatcher: it forwards to the callback module
  resolved from the provider struct.

  ## Implementations

  - `ExAgent.Providers.OpenAI`
  - `ExAgent.Providers.Gemini`
  - `ExAgent.Providers.DeepSeek`

  ## Extensibility

  To add a new provider, define a struct and implement this behaviour:

      defmodule MyApp.Providers.CustomLLM do
        @behaviour ExAgent.Provider
        defstruct [:api_key, :model, :base_url, :system_prompt, :req, tools: []]

        @impl true
        def chat(provider, messages, opts) do
          # Your implementation here
        end
      end
  """

  @doc """
  Sends a list of messages to the LLM and returns the assistant's response.

  Returns `{:ok, message}` for a regular response, `{:tool_call, name, args}`
  when the LLM wants to invoke a tool, or `{:error, reason}` on failure.
  """
  @callback chat(provider :: struct(), [ExAgent.Message.t()], keyword()) ::
              {:ok, ExAgent.Message.t()}
              | {:tool_call, String.t(), map()}
              | {:error, term()}

  @doc """
  Uploads binary file data to the provider and returns a file reference.

  Optional — providers without file support (e.g. DeepSeek) omit this callback.
  """
  @callback upload(provider :: struct(), binary(), String.t(), keyword()) ::
              {:ok, ExAgent.FileRef.t()} | {:error, term()}

  @doc """
  Streams the assistant's response as a lazy enumerable of text chunks.

  Optional — providers without streaming support omit this callback.
  """
  @callback stream(provider :: struct(), [ExAgent.Message.t()], keyword()) :: Enumerable.t()

  @optional_callbacks upload: 4, stream: 3

  @doc """
  Dispatches a chat request to the provider's implementation.
  """
  @spec chat(struct(), [ExAgent.Message.t()], keyword()) ::
          {:ok, ExAgent.Message.t()}
          | {:tool_call, String.t(), map()}
          | {:error, term()}
  def chat(provider, messages, opts \\ []) do
    provider.__struct__.chat(provider, messages, opts)
  end

  @doc """
  Dispatches a file upload to the provider's implementation.

  Returns `{:error, {:unsupported, :upload, module}}` if the provider does not
  implement `upload/4`.
  """
  @spec upload(struct(), binary(), String.t(), keyword()) ::
          {:ok, ExAgent.FileRef.t()} | {:error, term()}
  def upload(provider, file_data, mime_type, opts \\ []) do
    mod = provider.__struct__

    if function_exported?(mod, :upload, 4) do
      mod.upload(provider, file_data, mime_type, opts)
    else
      {:error, {:unsupported, :upload, mod}}
    end
  end

  @doc """
  Dispatches a streaming chat request to the provider's implementation.

  Returns a lazy enumerable of text chunks. Raises `ArgumentError` if the
  provider does not implement `stream/3`.
  """
  @spec stream(struct(), [ExAgent.Message.t()], keyword()) :: Enumerable.t()
  def stream(provider, messages, opts \\ []) do
    mod = provider.__struct__

    if function_exported?(mod, :stream, 3) do
      mod.stream(provider, messages, opts)
    else
      raise ArgumentError, "#{inspect(mod)} does not support streaming (stream/3)"
    end
  end
end
