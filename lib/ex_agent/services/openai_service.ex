defmodule ExAgent.Services.OpenAIService do
  @moduledoc """
  HTTP service for OpenAI chat completions API.

  Handles payload formatting and response parsing for the
  OpenAI `/chat/completions` endpoint using Req.
  """

  alias ExAgent.Providers.OpenAI
  alias ExAgent.Services.{OpenAIDialect, OpenAIUploadService, Streaming}
  alias ExAgent.{Attachment, Error, FileRef, Message, UploadCache}

  # Above this, bytes go through the Files API instead of a base64 data URI.
  @max_inline_bytes 20_000_000

  @chat_opts_schema [
    temperature: [type: {:or, [:float, :integer, nil]}],
    max_tokens: [type: {:or, [:pos_integer, nil]}],
    tool_choice: [type: {:or, [:string, :map]}, default: "auto"],
    built_in_tools: [type: {:list, {:or, [:atom, :map]}}, default: []]
  ]

  @doc """
  Sends a chat completion request to the OpenAI API.
  """
  @spec chat(
          OpenAI.t(),
          [Message.t()],
          keyword()
        ) ::
          {:ok, Response.t()}
          | {:tool_calls, [map()]}
          | {:error, Error.t()}
  def chat(%OpenAI{} = provider, messages, opts \\ []) do
    with {:ok, messages} <- prepare_attachments(provider, messages) do
      do_chat(provider, messages, opts)
    end
  end

  @spec do_chat(OpenAI.t(), [Message.t()], keyword()) ::
          {:ok, Response.t()}
          | {:tool_calls, [map()]}
          | {:error, Error.t()}
  defp do_chat(provider, messages, opts) do
    opts = prepare_opts(provider, opts)
    body = build_chat_body(provider, messages, opts)

    Req.post(provider.req,
      url: "/chat/completions",
      json: body,
      connect_options: [timeout: :timer.minutes(5)],
      receive_timeout: :timer.minutes(5)
    )
    |> Error.from_result(OpenAI)
    |> case do
      {:ok, response_body} -> parse_response(response_body)
      {:error, _error} = failure -> failure
    end
  end

  @doc """
  Streams a chat completion from the OpenAI API as a lazy enumerable of text chunks.
  """
  @spec stream(OpenAI.t(), [Message.t()], keyword()) :: Enumerable.t()
  def stream(%OpenAI{} = provider, messages, opts \\ []) do
    messages =
      case prepare_attachments(provider, messages) do
        {:ok, prepared} -> prepared
        {:error, error} -> raise error
      end

    opts = prepare_opts(provider, opts)

    body =
      provider
      |> build_chat_body(messages, opts)
      |> Map.put("stream", true)
      |> Map.put("stream_options", %{"include_usage" => true})

    Streaming.stream(
      provider.req,
      [url: "/chat/completions", json: body, receive_timeout: :timer.minutes(5)],
      OpenAI,
      &OpenAIDialect.chunks/1
    )
  end

  # Decides how each attachment reaches OpenAI, before any body building:
  #
  #   * `:url`      -> referenced as `image_url` / `file_url`, never fetched
  #   * `:file_ref` -> already uploaded, referenced by id
  #   * under the inline ceiling -> bytes loaded and sent as a base64 data URI
  #   * over it     -> uploaded through the Files API (cached by content hash)
  #
  # Leaving this out of `format_attachment/1` keeps that function a pure
  # formatter with no IO.
  @spec prepare_attachments(OpenAI.t(), [Message.t()]) ::
          {:ok, [Message.t()]} | {:error, Error.t()}
  defp prepare_attachments(provider, messages),
    do: Message.map_attachments(messages, &place(provider, &1))

  @spec place(OpenAI.t(), Attachment.t()) :: {:ok, Attachment.t()} | {:error, Error.t()}
  defp place(_provider, %Attachment{kind: :url} = attachment), do: {:ok, attachment}

  # Chat completions has no way to reference an uploaded *image*: `image_file`
  # is the Assistants API's shape, `image_url` demands a real URL, and the `file`
  # part only accepts PDFs. Verified against the live API - every variant 400s.
  defp place(_provider, %Attachment{kind: :file_ref, modality: :image}) do
    {:error,
     Error.new(
       :unsupported,
       "OpenAI chat completions cannot reference an uploaded image; pass the image " <>
         "as %{path: ...}, %{data: ...}, or a public %{url: ...} instead",
       OpenAI
     )}
  end

  defp place(_provider, %Attachment{kind: :file_ref} = attachment), do: {:ok, attachment}

  # For the same reason an oversized image is refused rather than uploaded: the
  # upload would succeed and the chat request that follows would always fail.
  defp place(_provider, %Attachment{modality: :image, byte_size: size} = attachment)
       when is_integer(size) and size > @max_inline_bytes do
    {:error,
     Error.new(
       :unsupported,
       "#{attachment.filename || "image"} is #{size} bytes, over OpenAI's " <>
         "#{@max_inline_bytes}-byte inline limit for images; resize it or host it at a URL",
       OpenAI
     )}
  end

  defp place(_provider, %Attachment{byte_size: size} = attachment)
       when is_nil(size) or size <= @max_inline_bytes,
       do: Attachment.load(attachment)

  defp place(provider, %Attachment{} = attachment) do
    with {:ok, data} <- Attachment.bytes(attachment) do
      scope = UploadCache.scope(OpenAI, provider.base_url, provider.api_key)

      case cached(provider, scope, data) do
        {:ok, ref} ->
          {:ok, as_file_ref(attachment, ref)}

        :miss ->
          upload_opts = [filename: attachment.filename || "upload"]

          with {:ok, ref} <-
                 OpenAIUploadService.upload(
                   provider.req,
                   data,
                   attachment.mime_type,
                   upload_opts
                 ) do
            if provider.upload_cache, do: UploadCache.put(scope, data, ref)
            {:ok, as_file_ref(attachment, ref)}
          end
      end
    end
  end

  @spec cached(OpenAI.t(), binary(), binary()) :: {:ok, FileRef.t()} | :miss
  defp cached(%OpenAI{upload_cache: false}, _scope, _data), do: :miss
  defp cached(_provider, scope, data), do: UploadCache.fetch(scope, data)

  # Drop the bytes once uploaded so a large payload is not retained in history.
  @spec as_file_ref(Attachment.t(), FileRef.t()) :: Attachment.t()
  defp as_file_ref(attachment, ref),
    do: %{attachment | kind: :file_ref, file_ref: ref, data: nil}

  @spec prepare_opts(OpenAI.t(), keyword()) :: keyword()
  defp prepare_opts(provider, opts) do
    max_tokens = opts[:max_tokens] || provider.max_tokens
    temperature = opts[:temperature] || provider.temperature

    opts
    |> Keyword.take(Keyword.keys(@chat_opts_schema))
    |> NimbleOptions.validate!(@chat_opts_schema)
    |> Keyword.merge(temperature: temperature, max_tokens: max_tokens)
  end

  @spec build_chat_body(OpenAI.t(), [Message.t()], keyword()) :: map()
  defp build_chat_body(provider, messages, opts) do
    OpenAIDialect.build_body(provider.model, messages,
      system_prompt: provider.system_prompt,
      tools: provider.tools,
      tool_choice: opts[:tool_choice],
      temperature: opts[:temperature],
      max_tokens: opts[:max_tokens],
      attachment_formatter: &format_attachment/1
    )
    |> maybe_add_built_in_tools(opts[:built_in_tools])
  end

  defp maybe_add_built_in_tools(body, []), do: body

  defp maybe_add_built_in_tools(body, built_in_tools) do
    Enum.reduce(built_in_tools, body, fn
      :web_search, acc ->
        Map.put(acc, "web_search_options", %{})

      %{web_search: location_opts}, acc ->
        user_location = %{"type" => "approximate", "approximate" => location_opts}
        Map.put(acc, "web_search_options", %{"user_location" => user_location})

      _other, acc ->
        acc
    end)
  end

  @spec format_attachment(map()) :: map()
  # Images never reach here - `place/2` refuses an uploaded image up front.
  defp format_attachment(%{file_ref: %FileRef{provider: :openai, file_id: fid}}),
    do: %{"type" => "file", "file" => %{"file_id" => fid}}

  defp format_attachment(%{kind: :url, url: url, mime_type: mime_type} = att) do
    if String.starts_with?(mime_type, "image/") do
      %{"type" => "image_url", "image_url" => %{"url" => url}}
    else
      file = %{"file_url" => url}
      file = if att.filename, do: Map.put(file, "filename", att.filename), else: file
      %{"type" => "file", "file" => file}
    end
  end

  defp format_attachment(%{data: data, mime_type: mime_type} = att) when is_binary(data) do
    if String.starts_with?(mime_type, "image/") do
      %{
        "type" => "image_url",
        "image_url" => %{"url" => "data:#{mime_type};base64,#{Base.encode64(data)}"}
      }
    else
      filename = Map.get(att, :filename) || "upload"

      %{
        "type" => "file",
        "file" => %{
          "filename" => filename,
          "file_data" => "data:#{mime_type};base64,#{Base.encode64(data)}"
        }
      }
    end
  end

  @spec parse_response(map()) ::
          {:ok, Response.t()}
          | {:tool_calls, [map()]}
          | {:error, Error.t()}
  defp parse_response(body), do: OpenAIDialect.parse_response(body, OpenAI)
end
