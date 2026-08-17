defmodule ExAgent.Services.OpenAICompatibleService do
  @moduledoc """
  HTTP service for OpenAI-compatible chat endpoints (vLLM, OpenRouter, Together,
  Groq, ...).

  Request and response shaping is shared with the other dialect speakers via an
  internal `OpenAIDialect` helper. What differs here is attachment delivery:
  these endpoints have no Files API, so bytes always become `data:` URIs and
  media is carried in `image_url` / `video_url` / `audio_url` content parts.
  """

  alias ExAgent.Providers.OpenAICompatible
  alias ExAgent.Services.{OpenAIDialect, Streaming, Structured}
  alias ExAgent.{Attachment, Error, Message, Source}

  @chat_opts_schema [
    temperature: [type: {:or, [:float, :integer, nil]}],
    max_tokens: [type: {:or, [:pos_integer, nil]}],
    tool_choice: [type: {:or, [:string, :map]}, default: "auto"],
    schema: [type: :any],
    schema_doc: [type: {:or, [:string, nil]}],
    receive_timeout: [type: {:or, [:pos_integer, nil]}]
  ]

  @doc """
  Sends a chat completion request to an OpenAI-compatible endpoint.
  """
  @spec chat(OpenAICompatible.t(), [Message.t()], keyword()) ::
          {:ok, Response.t()}
          | {:tool_calls, [map()]}
          | {:error, Error.t()}
  def chat(%OpenAICompatible{} = provider, messages, opts \\ []) do
    with {:ok, messages} <- prepare_attachments(provider, messages) do
      opts = prepare_opts(provider, opts)

      with {:ok, format} <- response_format(opts) do
        provider.req
        |> Req.post(
          url: "/chat/completions",
          json: build_chat_body(provider, messages, opts, format),
          connect_options: [timeout: opts[:receive_timeout]],
          receive_timeout: opts[:receive_timeout]
        )
        |> Error.from_result(OpenAICompatible)
        |> case do
          {:ok, response_body} ->
            response_body
            |> OpenAIDialect.parse_response(OpenAICompatible)
            |> Structured.decode(opts[:schema], OpenAICompatible)

          {:error, _error} = failure ->
            failure
        end
      end
    end
  end

  # `response_format` and not `guided_json`: vLLM accepts the latter, ignores it,
  # and answers with prose - verified against a running deployment. A server that
  # ignores `response_format` too is caught when the cast fails.
  @spec response_format(keyword()) :: {:ok, map() | nil} | {:error, Error.t()}
  defp response_format(opts) do
    with {:ok, json_schema} <- Structured.json_schema(opts[:schema], :openai, opts) do
      case json_schema do
        nil ->
          {:ok, nil}

        schema ->
          {:ok,
           %{
             "type" => "json_schema",
             "json_schema" => %{
               "name" => Structured.name(opts[:schema]),
               "strict" => true,
               "schema" => schema
             }
           }}
      end
    end
  end

  @doc """
  Streams a chat completion as a lazy enumerable of text chunks.
  """
  @spec stream(OpenAICompatible.t(), [Message.t()], keyword()) :: Enumerable.t()
  def stream(%OpenAICompatible{} = provider, messages, opts \\ []) do
    messages =
      case prepare_attachments(provider, messages) do
        {:ok, prepared} -> prepared
        {:error, error} -> raise error
      end

    opts = prepare_opts(provider, opts)

    format =
      case response_format(opts) do
        {:ok, format} -> format
        {:error, error} -> raise error
      end

    body =
      provider
      |> build_chat_body(messages, opts, format)
      |> Map.put("stream", true)
      |> Map.put("stream_options", %{"include_usage" => true})

    Streaming.stream(
      provider.req,
      [url: "/chat/completions", json: body, receive_timeout: opts[:receive_timeout]],
      OpenAICompatible,
      &OpenAIDialect.chunks/1
    )
  end

  @spec build_chat_body(OpenAICompatible.t(), [Message.t()], keyword(), map() | nil) :: map()
  defp build_chat_body(provider, messages, opts, response_format) do
    OpenAIDialect.build_body(provider.model, messages,
      system_prompt: provider.system_prompt,
      tools: provider.tools,
      tool_choice: opts[:tool_choice],
      temperature: opts[:temperature],
      max_tokens: opts[:max_tokens],
      response_format: response_format,
      attachment_formatter: &format_attachment/1
    )
  end

  @spec prepare_opts(OpenAICompatible.t(), keyword()) :: keyword()
  defp prepare_opts(provider, opts) do
    max_tokens = opts[:max_tokens] || provider.max_tokens
    temperature = opts[:temperature] || provider.temperature
    receive_timeout = opts[:receive_timeout] || provider.receive_timeout

    opts
    |> Keyword.take(Keyword.keys(@chat_opts_schema))
    |> NimbleOptions.validate!(@chat_opts_schema)
    |> Keyword.merge(
      temperature: temperature,
      max_tokens: max_tokens,
      receive_timeout: receive_timeout
    )
  end

  # There is no Files API here, so the only decisions are "reference the URL" or
  # "inline the bytes" - and refusing anything too large to inline, rather than
  # truncating it or retrying forever.
  @spec prepare_attachments(OpenAICompatible.t(), [Message.t()]) ::
          {:ok, [Message.t()]} | {:error, Error.t()}
  defp prepare_attachments(provider, messages),
    do: Message.map_attachments(messages, &place(provider, &1))

  @spec place(OpenAICompatible.t(), Attachment.t()) ::
          {:ok, Attachment.t()} | {:error, Error.t()}
  defp place(_provider, %Attachment{kind: :url} = attachment), do: {:ok, attachment}

  defp place(_provider, %Attachment{kind: :file_ref}) do
    {:error,
     Error.new(
       :unsupported,
       "OpenAI-compatible endpoints have no Files API, so an uploaded file " <>
         "reference cannot be used; pass :url, :path, or :data instead",
       OpenAICompatible
     )}
  end

  defp place(provider, %Attachment{byte_size: size} = attachment)
       when is_integer(size) and size > 0 do
    if size > provider.max_inline_bytes do
      {:error, too_large(provider, attachment, size)}
    else
      Attachment.load(attachment)
    end
  end

  defp place(_provider, %Attachment{} = attachment), do: Attachment.load(attachment)

  @spec too_large(OpenAICompatible.t(), Attachment.t(), non_neg_integer()) :: Error.t()
  defp too_large(provider, attachment, size) do
    Error.new(
      :unsupported,
      "#{attachment.filename || "attachment"} is #{size} bytes, over :max_inline_bytes " <>
        "(#{provider.max_inline_bytes}). This endpoint has no Files API - host the file " <>
        "at a URL the container can reach and pass it as %{url: ...}",
      OpenAICompatible
    )
  end

  # Media parts follow the `{"type" => "<kind>_url", "<kind>_url" => %{"url" => ...}}`
  # shape, where the URL may be a `data:` URI. Documents use the dialect's `file`
  # part instead. Whether a served model accepts any of them is model-dependent,
  # which is why `:modalities` must be declared per deployment.
  @spec format_attachment(map()) :: map()
  defp format_attachment(%{modality: :document, kind: :url, url: url} = attachment) do
    file = %{"file_url" => url}
    file = if attachment.filename, do: Map.put(file, "filename", attachment.filename), else: file

    %{"type" => "file", "file" => file}
    |> merge_provider_opts(attachment)
  end

  defp format_attachment(%{modality: :document} = attachment) do
    # `filename` is required alongside `file_data`; gateways reject the part
    # without it, so fall back rather than omitting the key.
    filename = Map.get(attachment, :filename) || "upload"

    %{
      "type" => "file",
      "file" => %{"filename" => filename, "file_data" => attachment_url(attachment)}
    }
    |> merge_provider_opts(attachment)
  end

  defp format_attachment(%{modality: modality} = attachment) do
    key = part_key(modality)

    %{"type" => key, key => %{"url" => attachment_url(attachment)}}
    |> merge_provider_opts(attachment)
  end

  @spec part_key(Source.modality()) :: String.t()
  defp part_key(:video), do: "video_url"
  defp part_key(:audio), do: "audio_url"
  defp part_key(_modality), do: "image_url"

  @spec attachment_url(map()) :: String.t()
  defp attachment_url(%{kind: :url, url: url}), do: url

  defp attachment_url(%{data: data, mime_type: mime_type}) when is_binary(data),
    do: "data:#{mime_type};base64,#{Base.encode64(data)}"

  # `:fps` / `:max_frames` and any string-keyed `:provider_opts` are merged into
  # the content part verbatim. Support is model-dependent - vLLM forwards extra
  # fields, but the served model decides whether they mean anything.
  @spec merge_provider_opts(map(), map()) :: map()
  defp merge_provider_opts(part, attachment) do
    opts = Map.get(attachment, :provider_opts) || %{}

    Enum.reduce(opts, part, fn
      {:fps, value}, acc -> Map.put(acc, "fps", value)
      {:max_frames, value}, acc -> Map.put(acc, "max_frames", value)
      {key, value}, acc when is_binary(key) -> Map.put(acc, key, value)
      _other, acc -> acc
    end)
  end
end
