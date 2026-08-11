defmodule ExAgent.Services.OpenAIDialect do
  @moduledoc false

  # Shared request/response shaping for providers that speak the OpenAI
  # chat-completions dialect (OpenAI and any OpenAI-compatible
  # endpoint).
  #
  # This is a service-layer helper, not a provider abstraction: each service
  # still owns its own HTTP call, option schema, built-in tool vocabulary, and
  # attachment content-part shape. Only the parts that are byte-identical across
  # providers live here.
  #
  # Attachment formatting is injected via `:attachment_formatter` because it is
  # the one piece that genuinely differs. Omitting it drops attachments from the
  # body, which is correct for text-only providers — they reject attachments at
  # the modality gate in `ExAgent.Provider` long before a body is built.

  alias ExAgent.{Chunk, Error, Message, Response, Tool}

  @type attachment_formatter :: (map() -> map())

  @type body_opts :: [
          system_prompt: String.t() | nil,
          tools: [Tool.t()],
          tool_choice: String.t() | map(),
          temperature: float() | nil,
          max_tokens: pos_integer() | nil,
          attachment_formatter: attachment_formatter() | nil
        ]

  @doc false
  @spec build_body(String.t(), [Message.t()], body_opts()) :: map()
  def build_body(model, messages, opts) do
    formatter = Keyword.get(opts, :attachment_formatter)

    %{"model" => model, "messages" => build_messages(messages, opts[:system_prompt], formatter)}
    |> maybe_put("temperature", opts[:temperature])
    |> maybe_put("max_tokens", opts[:max_tokens])
    |> maybe_add_tools(opts[:tools] || [], opts[:tool_choice])
  end

  @doc false
  @spec parse_response(map(), module()) ::
          {:ok, Response.t()}
          | {:tool_calls, [map()]}
          | {:error, Error.t()}
  def parse_response(
        %{"choices" => [%{"message" => %{} = message} | _] = choices} = body,
        provider
      ) do
    choice = hd(choices)

    case tool_calls(message) do
      [] -> text_response(message, choice, body, provider)
      calls -> {:tool_calls, calls}
    end
  end

  def parse_response(body, provider), do: {:error, Error.unexpected_response(body, provider)}

  # Every requested call is returned. Taking only the first left the model
  # believing tools ran that never did.
  @spec tool_calls(map()) :: [map()]
  defp tool_calls(%{"tool_calls" => calls}) when is_list(calls) do
    for %{"function" => %{"name" => name} = function} = call <- calls do
      %{
        # The API's own id is what a tool result must be correlated by.
        "id" => Map.get(call, "id") || name,
        "name" => name,
        "args" => decode_arguments(Map.get(function, "arguments"))
      }
    end
  end

  defp tool_calls(_message), do: []

  # A message with neither content nor tool calls — a bare refusal, say — is an
  # unparseable shape, not a crash.
  @spec text_response(map(), map(), map(), module()) ::
          {:ok, Response.t()} | {:error, Error.t()}
  defp text_response(%{"content" => content} = message, choice, body, _provider)
       when is_binary(content) or is_nil(content) do
    assistant = %Message{role: :assistant, content: content || "", tool_calls: nil}

    {:ok,
     Response.new(assistant,
       thinking: message["reasoning_content"],
       usage: usage_map(body["usage"]),
       finish_reason: Chunk.finish_reason(choice["finish_reason"])
     )}
  end

  defp text_response(_message, _choice, body, provider),
    do: {:error, Error.unexpected_response(body, provider)}

  @spec usage_map(term()) :: map()
  defp usage_map(%{} = usage) do
    %{
      input_tokens: usage["prompt_tokens"],
      output_tokens: usage["completion_tokens"],
      total_tokens: usage["total_tokens"]
    }
  end

  defp usage_map(_usage), do: %{}

  @doc false
  # Maps one chat-completions stream frame onto chunks. Shared by every dialect
  # speaker — a compatible endpoint's frames are identical to OpenAI's.
  @spec chunks(map()) :: [Chunk.t()]
  def chunks(%{"choices" => [choice | _]} = frame) do
    delta = Map.get(choice, "delta") || %{}

    text_chunks(delta) ++
      tool_call_chunks(delta) ++
      usage_chunks(frame) ++
      finish_chunks(choice)
  end

  def chunks(%{"usage" => usage} = frame) when is_map(usage), do: usage_chunks(frame)
  def chunks(_frame), do: []

  @spec text_chunks(map()) :: [Chunk.t()]
  defp text_chunks(delta) do
    content = Map.get(delta, "content")
    # Reasoning models (and some vLLM builds) put the reasoning trace here.
    reasoning = Map.get(delta, "reasoning_content")

    maybe_text(&Chunk.thinking_delta/1, reasoning) ++ maybe_text(&Chunk.text_delta/1, content)
  end

  @spec maybe_text((String.t() -> Chunk.t()), term()) :: [Chunk.t()]
  defp maybe_text(builder, text) when is_binary(text) and text != "", do: [builder.(text)]
  defp maybe_text(_builder, _text), do: []

  # Arguments arrive as JSON fragments to be concatenated by index; a chunk
  # carries only the fragment it saw.
  @spec tool_call_chunks(map()) :: [Chunk.t()]
  defp tool_call_chunks(%{"tool_calls" => tool_calls}) when is_list(tool_calls) do
    Enum.map(tool_calls, fn tool_call ->
      function = Map.get(tool_call, "function") || %{}

      %Chunk{
        type: :tool_call_delta,
        index: Map.get(tool_call, "index", 0),
        id: Map.get(tool_call, "id"),
        name: Map.get(function, "name"),
        arguments: Map.get(function, "arguments")
      }
    end)
  end

  defp tool_call_chunks(_delta), do: []

  @spec usage_chunks(map()) :: [Chunk.t()]
  defp usage_chunks(%{"usage" => usage}) when is_map(usage) do
    [
      Chunk.usage(%{
        input_tokens: usage["prompt_tokens"],
        output_tokens: usage["completion_tokens"],
        total_tokens: usage["total_tokens"]
      })
    ]
  end

  defp usage_chunks(_frame), do: []

  @spec finish_chunks(map()) :: [Chunk.t()]
  defp finish_chunks(%{"finish_reason" => reason}) when is_binary(reason),
    do: [Chunk.done(Chunk.finish_reason(reason))]

  defp finish_chunks(_choice), do: []

  @spec build_messages([Message.t()], String.t() | nil, attachment_formatter() | nil) :: [map()]
  defp build_messages(messages, system_prompt, formatter) do
    formatted = Enum.map(messages, &format_message(&1, formatter))

    case system_prompt do
      nil -> formatted
      prompt -> [%{"role" => "system", "content" => prompt} | formatted]
    end
  end

  @spec format_message(Message.t(), attachment_formatter() | nil) :: map()
  defp format_message(
         %Message{role: :user, content: content, attachments: attachments},
         formatter
       )
       when is_list(attachments) and attachments != [] and is_function(formatter, 1) do
    file_parts = Enum.map(attachments, formatter)

    %{"role" => "user", "content" => file_parts ++ [%{"type" => "text", "text" => content}]}
  end

  defp format_message(
         %Message{role: :assistant, content: content, tool_calls: tool_calls},
         _formatter
       )
       when is_list(tool_calls) and tool_calls != [] do
    %{
      "role" => "assistant",
      "content" => content,
      "tool_calls" => Enum.map(tool_calls, &format_tool_call/1)
    }
  end

  defp format_message(
         %Message{role: :tool, content: content, tool_call_id: tool_call_id},
         _formatter
       ) do
    %{"role" => "tool", "content" => content, "tool_call_id" => tool_call_id}
  end

  defp format_message(%Message{role: role, content: content}, _formatter) do
    %{"role" => to_string(role), "content" => content}
  end

  # The id the API issued, so a tool result correlates back to the right call —
  # the name is only a fallback for a provider that has none, and it collides as
  # soon as the model calls one tool twice in a turn.
  @spec format_tool_call(map()) :: map()
  defp format_tool_call(tool_call) do
    %{
      "id" => tool_call["id"] || tool_call["name"],
      "type" => "function",
      "function" => %{
        "name" => tool_call["name"],
        "arguments" => Jason.encode!(tool_call["args"] || %{})
      }
    }
  end

  @spec maybe_put(map(), String.t(), term()) :: map()
  defp maybe_put(body, _key, nil), do: body
  defp maybe_put(body, key, value), do: Map.put(body, key, value)

  @spec maybe_add_tools(map(), [Tool.t()], String.t() | map() | nil) :: map()
  defp maybe_add_tools(body, [], _choice), do: body

  defp maybe_add_tools(body, tools, choice) do
    body
    |> Map.put("tools", Enum.map(tools, &format_tool/1))
    |> Map.put("tool_choice", choice)
  end

  @spec format_tool(Tool.t()) :: map()
  defp format_tool(%Tool{name: name, description: description, parameters: parameters}) do
    %{
      "type" => "function",
      "function" => %{
        "name" => name,
        "description" => description,
        "parameters" => parameters
      }
    }
  end

  @spec decode_arguments(String.t() | nil) :: map()
  defp decode_arguments(args) when is_binary(args) do
    case Jason.decode(args) do
      {:ok, decoded} when is_map(decoded) -> decoded
      _other -> %{"raw" => args}
    end
  end

  defp decode_arguments(_args), do: %{}
end
