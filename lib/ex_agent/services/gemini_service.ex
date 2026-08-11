defmodule ExAgent.Services.GeminiService do
  @moduledoc """
  HTTP service for Google Gemini API.

  Handles payload formatting and response parsing for the
  Gemini `generateContent` endpoint using Req.
  """

  alias ExAgent.{Attachment, Chunk, Error, FileRef, Message, Response, UploadCache}
  alias ExAgent.Providers.Gemini
  alias ExAgent.Services.{GeminiUploadService, Streaming}

  # Gemini's inline request-size ceiling. PDFs get a larger allowance; anything
  # above goes through the Files API instead.
  @max_inline_bytes 20_000_000
  @max_inline_pdf_bytes 50_000_000

  @chat_opts_schema [
    temperature: [type: :float, default: 0.7],
    max_output_tokens: [type: :pos_integer],
    built_in_tools: [type: {:list, :atom}, default: []]
  ]

  @gemini_built_in_tools %{
    google_search: %{"google_search" => %{}},
    code_execution: %{"code_execution" => %{}},
    url_context: %{"url_context" => %{}}
  }

  @doc """
  Sends a chat completion request to the Gemini API.
  """
  @spec chat(
          Gemini.t(),
          [Message.t()],
          keyword()
        ) ::
          {:ok, Response.t()} | {:tool_call, String.t(), map()} | {:error, Error.t()}
  def chat(%Gemini{} = provider, messages, opts \\ []) do
    with {:ok, messages} <- prepare_attachments(provider, messages, opts) do
      opts = prepare_opts(provider, opts)
      body = build_chat_body(messages, provider.tools, provider.system_prompt, opts)

      Req.post(provider.req,
        url: "/models/#{provider.model}:generateContent",
        json: body,
        connect_options: [timeout: :timer.minutes(5)],
        receive_timeout: :timer.minutes(5)
      )
      |> Error.from_result(Gemini)
      |> case do
        {:ok, response_body} -> parse_response(response_body)
        {:error, _error} = failure -> failure
      end
    end
  end

  @doc """
  Streams a chat completion from the Gemini API as a lazy enumerable of text chunks.
  """
  @spec stream(Gemini.t(), [Message.t()], keyword()) :: Enumerable.t()
  def stream(%Gemini{} = provider, messages, opts \\ []) do
    # A lazy enumerable cannot carry an error tuple at construction time, so
    # attachment preparation failures surface as a raise here, matching
    # `ExAgent.Provider.stream/3`.
    messages =
      case prepare_attachments(provider, messages, opts) do
        {:ok, prepared} -> prepared
        {:error, error} -> raise error
      end

    opts = prepare_opts(provider, opts)
    body = build_chat_body(messages, provider.tools, provider.system_prompt, opts)

    Streaming.stream(
      provider.req,
      [
        url: "/models/#{provider.model}:streamGenerateContent?alt=sse",
        json: body,
        receive_timeout: :timer.minutes(5)
      ],
      Gemini,
      &chunks/1
    )
  end

  # Decides how each attachment reaches Gemini, before any body building:
  #
  #   * `:url`      -> referenced as-is; Gemini accepts public https URLs and
  #                    Files API URIs in the same field, so nothing is fetched
  #   * `:file_ref` -> already uploaded, referenced as-is
  #   * under the inline ceiling -> bytes loaded and base64-inlined
  #   * over it     -> uploaded through the Files API (cached by content hash)
  #
  # Leaving this out of `format_attachment/1` keeps that function a pure
  # formatter with no IO.
  @spec prepare_attachments(Gemini.t(), [Message.t()], keyword()) ::
          {:ok, [Message.t()]} | {:error, Error.t()}
  defp prepare_attachments(provider, messages, opts),
    do: Message.map_attachments(messages, &place(provider, &1, opts))

  @spec place(Gemini.t(), Attachment.t(), keyword()) ::
          {:ok, Attachment.t()} | {:error, Error.t()}
  defp place(_provider, %Attachment{kind: kind} = attachment, _opts)
       when kind in [:url, :file_ref],
       do: {:ok, attachment}

  defp place(provider, %Attachment{} = attachment, opts) do
    if inlineable?(attachment) do
      Attachment.load(attachment)
    else
      upload(provider, attachment, opts)
    end
  end

  @spec inlineable?(Attachment.t()) :: boolean()
  defp inlineable?(%Attachment{byte_size: nil}), do: true

  defp inlineable?(%Attachment{byte_size: size, mime_type: "application/pdf"}),
    do: size <= @max_inline_pdf_bytes

  defp inlineable?(%Attachment{byte_size: size}), do: size <= @max_inline_bytes

  @spec upload(Gemini.t(), Attachment.t(), keyword()) ::
          {:ok, Attachment.t()} | {:error, Error.t()}
  defp upload(provider, attachment, opts) do
    with {:ok, data} <- Attachment.bytes(attachment) do
      scope = UploadCache.scope(Gemini, provider.base_url, provider.api_key)

      case cached(provider, scope, data) do
        {:ok, ref} ->
          {:ok, as_file_ref(attachment, ref)}

        :miss ->
          # Reuse the provider's Req client so its configuration (and, in tests,
          # its plug stub) applies. The upload service overrides the URL
          # absolutely, so the client's base_url is not consulted.
          # Polling cadence is tunable per request: a two-hour video needs a
          # longer cap than the ~2 minute default.
          upload_opts =
            opts
            |> Keyword.take([:poll_interval_ms, :max_poll_attempts])
            |> Keyword.merge(filename: attachment.filename || "upload", req: provider.req)

          with {:ok, ref} <-
                 GeminiUploadService.upload(
                   provider.api_key,
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

  @spec cached(Gemini.t(), binary(), binary()) :: {:ok, FileRef.t()} | :miss
  defp cached(%Gemini{upload_cache: false}, _scope, _data), do: :miss
  defp cached(_provider, scope, data), do: UploadCache.fetch(scope, data)

  # Drop the bytes once uploaded so a large payload is not retained in history.
  @spec as_file_ref(Attachment.t(), FileRef.t()) :: Attachment.t()
  defp as_file_ref(attachment, ref),
    do: %{attachment | kind: :file_ref, file_ref: ref, data: nil}

  @spec prepare_opts(Gemini.t(), keyword()) :: keyword()
  defp prepare_opts(provider, opts) do
    max_tokens = opts[:max_tokens] || provider.max_tokens
    temperature = opts[:temperature] || provider.temperature

    opts
    |> Keyword.take(Keyword.keys(@chat_opts_schema))
    |> NimbleOptions.validate!(@chat_opts_schema)
    |> Keyword.merge(temperature: temperature, max_tokens: max_tokens)
  end

  # Gemini's streaming envelope has changed between API generations, so match the
  # shapes we know and fall back to a recursive scan for text rather than
  # dropping content on the floor when Google renames something.
  @spec chunks(map()) :: [Chunk.t()]
  defp chunks(%{"candidates" => [candidate | _]} = frame) do
    parts = get_in(candidate, ["content", "parts"]) || []

    Enum.flat_map(parts, &part_chunks/1) ++
      usage_chunks(frame) ++
      finish_chunks(candidate)
  end

  defp chunks(%{"usageMetadata" => _} = frame), do: usage_chunks(frame)
  defp chunks(frame), do: fallback_text(frame)

  @spec part_chunks(map()) :: [Chunk.t()]
  defp part_chunks(%{"text" => text, "thought" => true}) when is_binary(text),
    do: [Chunk.thinking_delta(text)]

  defp part_chunks(%{"text" => text}) when is_binary(text) and text != "",
    do: [Chunk.text_delta(text)]

  # Gemini sends a function call whole rather than fragmented; emitting it as a
  # single complete "fragment" keeps the accumulate-by-index contract intact.
  defp part_chunks(%{"functionCall" => %{"name" => name} = call}) do
    [
      %Chunk{
        type: :tool_call_delta,
        index: 0,
        name: name,
        arguments: Jason.encode!(Map.get(call, "args") || %{})
      }
    ]
  end

  defp part_chunks(_part), do: []

  @spec usage_chunks(map()) :: [Chunk.t()]
  defp usage_chunks(%{"usageMetadata" => usage}) when is_map(usage) do
    [
      Chunk.usage(%{
        input_tokens: usage["promptTokenCount"],
        output_tokens: usage["candidatesTokenCount"],
        total_tokens: usage["totalTokenCount"]
      })
    ]
  end

  defp usage_chunks(_frame), do: []

  @spec finish_chunks(map()) :: [Chunk.t()]
  defp finish_chunks(%{"finishReason" => reason}) when is_binary(reason),
    do: [Chunk.done(Chunk.finish_reason(reason))]

  defp finish_chunks(_candidate), do: []

  # Last resort: depth-first scan for any "text" value, so a renamed envelope
  # degrades to plain text instead of silence.
  @spec fallback_text(term()) :: [Chunk.t()]
  defp fallback_text(%{"text" => text}) when is_binary(text) and text != "",
    do: [Chunk.text_delta(text)]

  defp fallback_text(map) when is_map(map),
    do: map |> Map.values() |> Enum.flat_map(&fallback_text/1)

  defp fallback_text(list) when is_list(list), do: Enum.flat_map(list, &fallback_text/1)
  defp fallback_text(_other), do: []

  @spec build_chat_body([Message.t()], [ExAgent.Tool.t()], String.t() | nil, keyword()) :: map()
  defp build_chat_body(messages, tools, system_prompt, opts) do
    %{"contents" => Enum.map(messages, &format_content/1)}
    |> maybe_add_system_instruction(system_prompt)
    |> maybe_add_tools(tools)
    |> maybe_add_built_in_tools(opts[:built_in_tools])
    |> maybe_add_generation_config(opts)
  end

  @spec format_content(Message.t()) :: map()
  defp format_content(%Message{role: :user, content: content, attachments: attachments})
       when is_list(attachments) and attachments != [] do
    file_parts = Enum.map(attachments, &format_attachment/1)

    %{"role" => "user", "parts" => file_parts ++ [%{"text" => content}]}
  end

  defp format_content(%Message{role: :user, content: content}) do
    %{"role" => "user", "parts" => [%{"text" => content}]}
  end

  defp format_content(%Message{role: :assistant, tool_calls: tool_calls})
       when is_list(tool_calls) and tool_calls != [] do
    parts =
      Enum.map(tool_calls, fn tc ->
        %{"functionCall" => %{"name" => tc["name"], "args" => tc["args"] || %{}}}
      end)

    %{"role" => "model", "parts" => parts}
  end

  defp format_content(%Message{role: :assistant, content: content}) do
    %{"role" => "model", "parts" => [%{"text" => content}]}
  end

  defp format_content(%Message{role: :tool, content: content, tool_call_id: tool_call_id}) do
    %{
      "role" => "user",
      "parts" => [
        %{
          "functionResponse" => %{
            "name" => tool_call_id || "unknown",
            "response" => %{"result" => content}
          }
        }
      ]
    }
  end

  defp format_content(%Message{role: :system, content: content}) do
    %{"role" => "user", "parts" => [%{"text" => content}]}
  end

  @spec format_attachment(map()) :: map()
  defp format_attachment(
         %{file_ref: %FileRef{provider: :gemini, file_uri: uri, mime_type: mt}} = att
       ) do
    %{"file_data" => %{"file_uri" => uri, "mime_type" => mt}}
    |> maybe_add_video_metadata(att)
  end

  # Gemini accepts a public https URL and a Files API URI in the same field, so
  # a URL attachment needs no download.
  defp format_attachment(%{kind: :url, url: url, mime_type: mime_type} = att) do
    %{"file_data" => %{"file_uri" => url, "mime_type" => mime_type}}
    |> maybe_add_video_metadata(att)
  end

  defp format_attachment(%{data: data, mime_type: mime_type} = att) when is_binary(data) do
    %{"inline_data" => %{"mime_type" => mime_type, "data" => Base.encode64(data)}}
    |> maybe_add_video_metadata(att)
  end

  # `:fps` is the one video knob with a direct Gemini equivalent. Anything under
  # `:provider_opts` with a string key is merged verbatim as an escape hatch.
  @spec maybe_add_video_metadata(map(), map()) :: map()
  defp maybe_add_video_metadata(part, att) do
    opts = Map.get(att, :provider_opts) || %{}

    metadata =
      %{}
      |> maybe_put("fps", opts[:fps])
      |> Map.merge(for {k, v} <- opts, is_binary(k), into: %{}, do: {k, v})

    if metadata == %{}, do: part, else: Map.put(part, "video_metadata", metadata)
  end

  defp maybe_add_system_instruction(body, nil), do: body

  defp maybe_add_system_instruction(body, prompt) do
    Map.put(body, "system_instruction", %{"parts" => [%{"text" => prompt}]})
  end

  defp maybe_add_tools(body, []), do: body

  defp maybe_add_tools(body, tools) do
    Map.put(body, "tools", [
      %{
        "functionDeclarations" =>
          Enum.map(tools, fn %ExAgent.Tool{name: name, description: desc, parameters: params} ->
            %{"name" => name, "description" => desc, "parameters" => params}
          end)
      }
    ])
  end

  defp maybe_add_built_in_tools(body, []), do: body

  defp maybe_add_built_in_tools(body, built_in_tools) do
    entries =
      Enum.map(built_in_tools, fn tool_name ->
        Map.get(@gemini_built_in_tools, tool_name, %{to_string(tool_name) => %{}})
      end)

    existing = Map.get(body, "tools", [])
    Map.put(body, "tools", existing ++ entries)
  end

  defp maybe_add_generation_config(body, opts) do
    config =
      %{}
      |> maybe_put("temperature", opts[:temperature])
      |> maybe_put("maxOutputTokens", opts[:max_output_tokens])

    if config == %{}, do: body, else: Map.put(body, "generationConfig", config)
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  @spec parse_response(map()) ::
          {:ok, Response.t()} | {:tool_call, String.t(), map()} | {:error, Error.t()}
  defp parse_response(%{"candidates" => [candidate | _]} = body) do
    parts = get_in(candidate, ["content", "parts"]) || []

    case parts do
      [%{"functionCall" => %{"name" => name, "args" => args}} | _] ->
        {:tool_call, name, args}

      [%{"text" => text} | _] ->
        {:ok,
         Response.new(%Message{role: :assistant, content: text},
           usage: gemini_usage(body["usageMetadata"]),
           finish_reason: Chunk.finish_reason(candidate["finishReason"])
         )}

      _ ->
        {:error, Error.unexpected_response(parts, Gemini)}
    end
  end

  defp parse_response(body), do: {:error, Error.unexpected_response(body, Gemini)}

  @spec gemini_usage(term()) :: map()
  defp gemini_usage(%{} = usage) do
    %{
      input_tokens: usage["promptTokenCount"],
      output_tokens: usage["candidatesTokenCount"],
      total_tokens: usage["totalTokenCount"]
    }
  end

  defp gemini_usage(_usage), do: %{}
end
