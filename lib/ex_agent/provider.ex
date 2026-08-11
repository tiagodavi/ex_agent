defmodule ExAgent.Provider do
  @moduledoc """
  Behaviour for LLM providers.

  A provider is an interchangeable *service* (OpenAI, Gemini, a self-hosted
  vLLM container, ...)
  implementing a shared capability. Each provider is a struct module that
  declares `@behaviour ExAgent.Provider` and implements `chat/3` (required)
  and optionally `upload/4` and `stream/3`. The provider struct carries its own
  config (api key, model, cached Req client) and is passed as the first argument.

  Failures are returned as `{:error, %ExAgent.Error{}}`.

  This module also acts as the dispatcher: it forwards to the callback module
  resolved from the provider struct.

  ## Implementations

  - `ExAgent.Providers.OpenAI`
  - `ExAgent.Providers.Gemini`
  - `ExAgent.Providers.OpenAICompatible`

  ## Extensibility

  To add a new provider, define a struct and implement this behaviour.
  `ExAgent.Error.from_result/2` handles the status classification:

      defmodule MyApp.Providers.CustomLLM do
        @behaviour ExAgent.Provider
        defstruct [:api_key, :model, :base_url, :system_prompt, :req, tools: []]

        @impl true
        def chat(provider, messages, opts) do
          Req.post(provider.req, url: "/chat", json: build_body(messages, opts))
          |> ExAgent.Error.from_result(__MODULE__)
          |> case do
            {:ok, body} -> parse_response(body)
            {:error, _error} = failure -> failure
          end
        end
      end
  """

  @doc """
  Sends a list of messages to the LLM and returns the assistant's response.

  Returns `{:ok, %ExAgent.Response{}}` for a regular response, `{:tool_call, name, args}`
  when the LLM wants to invoke a tool, or `{:error, %ExAgent.Error{}}` on failure.
  """
  @callback chat(provider :: struct(), [ExAgent.Message.t()], keyword()) ::
              {:ok, ExAgent.Response.t()}
              | {:tool_call, String.t(), map()}
              | {:error, ExAgent.Error.t()}

  @doc """
  Uploads binary file data to the provider and returns a file reference.

  Optional — providers without a files API (e.g. `ExAgent.Providers.OpenAICompatible`)
  omit this callback.
  """
  @callback upload(provider :: struct(), binary(), String.t(), keyword()) ::
              {:ok, ExAgent.FileRef.t()} | {:error, ExAgent.Error.t()}

  @doc """
  Streams the assistant's response as a lazy enumerable of text chunks.

  Optional — providers without streaming support omit this callback.
  """
  @callback stream(provider :: struct(), [ExAgent.Message.t()], keyword()) :: Enumerable.t()

  @doc """
  Returns the attachment modalities this provider instance accepts.

  Optional — providers that omit it are treated as text-only, so an attachment
  they cannot handle fails loudly instead of being dropped on the floor.

  Takes the provider *struct*, not the module, because a provider pointed at a
  configurable backend declares its modalities per instance.
  """
  @callback supported_modalities(provider :: struct()) :: [ExAgent.Source.modality()]

  @doc """
  Generates embedding vectors for a list of inputs.

  Optional — providers without an embeddings endpoint omit this callback.
  """
  @callback embed(provider :: struct(), [ExAgent.Embeddings.input()], keyword()) ::
              {:ok, ExAgent.Embeddings.t()} | {:error, ExAgent.Error.t()}

  @optional_callbacks upload: 4, stream: 3, supported_modalities: 1, embed: 3

  @doc """
  Returns the attachment modalities the provider accepts.

  Defaults to `[:text]` for providers that do not implement
  `c:supported_modalities/1`.
  """
  @spec supported_modalities(struct()) :: [ExAgent.Source.modality()]
  def supported_modalities(provider) do
    mod = provider.__struct__

    if function_exported?(mod, :supported_modalities, 1) do
      mod.supported_modalities(provider)
    else
      [:text]
    end
  end

  @doc """
  Dispatches a chat request to the provider's implementation.

  Returns `{:error, %ExAgent.Error{type: :unsupported}}` if any attachment's
  modality is not in the provider's `c:supported_modalities/1`.
  """
  @spec chat(struct(), [ExAgent.Message.t()], keyword()) ::
          {:ok, ExAgent.Response.t()}
          | {:tool_call, String.t(), map()}
          | {:error, ExAgent.Error.t()}
  def chat(provider, messages, opts \\ []) do
    with :ok <- check_modalities(provider, messages) do
      provider.__struct__.chat(provider, messages, opts)
    end
  end

  @doc """
  Dispatches a file upload to the provider's implementation.

  Returns `{:error, %ExAgent.Error{type: :unsupported}}` if the provider does not
  implement `upload/4`.
  """
  @spec upload(struct(), binary(), String.t(), keyword()) ::
          {:ok, ExAgent.FileRef.t()} | {:error, ExAgent.Error.t()}
  def upload(provider, file_data, mime_type, opts \\ []) do
    mod = provider.__struct__

    if function_exported?(mod, :upload, 4) do
      mod.upload(provider, file_data, mime_type, opts)
    else
      {:error,
       ExAgent.Error.new(:unsupported, "#{inspect(mod)} does not support file uploads", mod)}
    end
  end

  @doc """
  Dispatches a streaming chat request to the provider's implementation.

  Returns a lazy enumerable of text chunks. Raises `ExAgent.Error` with
  `type: :unsupported` if the provider does not implement `stream/3`, or if any
  attachment's modality is unsupported — a lazy enumerable has nowhere to carry
  an error tuple at construction time.
  """
  @spec stream(struct(), [ExAgent.Message.t()], keyword()) :: Enumerable.t()
  def stream(provider, messages, opts \\ []) do
    mod = provider.__struct__

    case check_modalities(provider, messages) do
      {:error, error} -> raise error
      :ok -> :ok
    end

    if function_exported?(mod, :stream, 3) do
      mod.stream(provider, messages, opts)
    else
      raise ExAgent.Error.new(:unsupported, "#{inspect(mod)} does not support streaming", mod)
    end
  end

  @doc """
  Dispatches an embedding request to the provider's implementation.

  Returns `{:error, %ExAgent.Error{type: :unsupported}}` if the provider does not
  implement `embed/3`. Follows the `upload/4` tuple convention rather than
  `stream/3`'s raise: `embed/3` already returns a result tuple for every other
  failure, and raising for just one of them would force callers to write both a
  `case` and a `try`.
  """
  @spec embed(struct(), [ExAgent.Embeddings.input()], keyword()) ::
          {:ok, ExAgent.Embeddings.t()} | {:error, ExAgent.Error.t()}
  def embed(provider, inputs, opts \\ []) do
    mod = provider.__struct__

    if function_exported?(mod, :embed, 3) do
      mod.embed(provider, inputs, opts)
    else
      {:error,
       ExAgent.Error.new(:unsupported, "#{inspect(mod)} does not support embeddings", mod)}
    end
  end

  # Modality is a property of the attachment's mime type and the supported set is
  # a property of the provider struct, so this gate is provider-agnostic and lives
  # here rather than being repeated in every service.
  @spec check_modalities(struct(), [ExAgent.Message.t()]) :: :ok | {:error, ExAgent.Error.t()}
  defp check_modalities(provider, messages) do
    supported = supported_modalities(provider)

    messages
    |> Enum.flat_map(fn message -> Map.get(message, :attachments) || [] end)
    |> Enum.find_value(fn attachment ->
      case Map.get(attachment, :modality) do
        nil -> nil
        modality -> if modality in supported, do: nil, else: modality
      end
    end)
    |> case do
      nil -> :ok
      modality -> {:error, unsupported_modality(provider.__struct__, modality)}
    end
  end

  @spec unsupported_modality(module(), atom()) :: ExAgent.Error.t()
  defp unsupported_modality(mod, modality) do
    name = mod |> Module.split() |> List.last()

    ExAgent.Error.new(
      :unsupported,
      "#{name} does not accept #{inspect(modality)} input",
      mod
    )
  end
end
