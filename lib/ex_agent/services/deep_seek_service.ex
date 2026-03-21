defmodule ExAgent.Services.DeepSeekService do
  @moduledoc """
  HTTP service for DeepSeek API.

  DeepSeek uses an OpenAI-compatible API format, so this service
  mirrors the OpenAI service with DeepSeek-specific defaults.
  """

  alias ExAgent.Message

  @chat_opts_schema [
    temperature: [type: :float, default: 0.7],
    max_tokens: [type: :pos_integer],
    tool_choice: [type: {:or, [:string, :map]}, default: "auto"],
    built_in_tools: [type: {:list, :atom}, default: []]
  ]

  @doc """
  Sends a chat completion request to the DeepSeek API.
  """
  @spec chat(
          Req.Request.t(),
          String.t(),
          [Message.t()],
          [ExAgent.Tool.t()],
          String.t() | nil,
          keyword()
        ) ::
          {:ok, Message.t()} | {:tool_call, String.t(), map()} | {:error, term()}
  def chat(req, model, messages, tools, system_prompt, opts \\ []) do
    opts = NimbleOptions.validate!(opts, @chat_opts_schema)
    body = build_chat_body(model, messages, tools, system_prompt, opts)

    case Req.post(req, url: "/chat/completions", json: body) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        parse_response(body)

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, {status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec build_chat_body(
          String.t(),
          [Message.t()],
          [ExAgent.Tool.t()],
          String.t() | nil,
          keyword()
        ) :: map()
  defp build_chat_body(model, messages, tools, system_prompt, opts) do
    %{"model" => model, "messages" => build_messages(messages, system_prompt)}
    |> maybe_add_temperature(opts[:temperature])
    |> maybe_add_max_tokens(opts[:max_tokens])
    |> maybe_add_tools(tools, opts[:tool_choice])
    |> maybe_add_built_in_tools(opts[:built_in_tools])
  end

  @spec build_messages([Message.t()], String.t() | nil) :: [map()]
  defp build_messages(messages, nil), do: Enum.map(messages, &format_message/1)

  defp build_messages(messages, system_prompt) do
    [%{"role" => "system", "content" => system_prompt} | Enum.map(messages, &format_message/1)]
  end

  @spec format_message(Message.t()) :: map()
  defp format_message(%Message{role: :assistant, content: content, tool_calls: tool_calls})
       when is_list(tool_calls) and tool_calls != [] do
    %{
      "role" => "assistant",
      "content" => content,
      "tool_calls" =>
        Enum.map(tool_calls, fn tc ->
          %{
            "id" => tc["name"],
            "type" => "function",
            "function" => %{"name" => tc["name"], "arguments" => Jason.encode!(tc["args"] || %{})}
          }
        end)
    }
  end

  defp format_message(%Message{role: :tool, content: content, tool_call_id: tool_call_id}) do
    %{"role" => "tool", "content" => content, "tool_call_id" => tool_call_id}
  end

  defp format_message(%Message{role: role, content: content}) do
    %{"role" => to_string(role), "content" => content}
  end

  defp maybe_add_temperature(body, nil), do: body
  defp maybe_add_temperature(body, temp), do: Map.put(body, "temperature", temp)

  defp maybe_add_max_tokens(body, nil), do: body
  defp maybe_add_max_tokens(body, max), do: Map.put(body, "max_tokens", max)

  defp maybe_add_tools(body, [], _choice), do: body

  defp maybe_add_tools(body, tools, choice) do
    body
    |> Map.put("tools", Enum.map(tools, &format_tool/1))
    |> Map.put("tool_choice", choice)
  end

  defp maybe_add_built_in_tools(body, []), do: body

  defp maybe_add_built_in_tools(body, built_in_tools) do
    Enum.reduce(built_in_tools, body, fn
      :thinking, acc ->
        Map.put(acc, "thinking", %{"type" => "enabled"})

      _other, acc ->
        acc
    end)
  end

  @spec format_tool(ExAgent.Tool.t()) :: map()
  defp format_tool(%ExAgent.Tool{name: name, description: desc, parameters: params}) do
    %{
      "type" => "function",
      "function" => %{
        "name" => name,
        "description" => desc,
        "parameters" => params
      }
    }
  end

  @spec parse_response(map()) ::
          {:ok, Message.t()} | {:tool_call, String.t(), map()} | {:error, term()}
  defp parse_response(%{"choices" => [%{"message" => message} | _]}) do
    case message do
      %{"tool_calls" => [%{"function" => %{"name" => name, "arguments" => args}} | _]} ->
        parsed_args =
          case Jason.decode(args) do
            {:ok, decoded} -> decoded
            {:error, _} -> %{"raw" => args}
          end

        {:tool_call, name, parsed_args}

      %{"content" => content} ->
        {:ok,
         %Message{
           role: :assistant,
           content: content || "",
           tool_calls: message["tool_calls"]
         }}
    end
  end

  defp parse_response(body), do: {:error, {:unexpected_response, body}}
end
