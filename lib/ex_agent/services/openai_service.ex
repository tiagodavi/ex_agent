defmodule ExAgent.Services.OpenAIService do
  @moduledoc """
  HTTP service for OpenAI chat completions API.

  Handles payload formatting and response parsing for the
  OpenAI `/chat/completions` endpoint using Req.
  """

  alias ExAgent.Providers.OpenAI
  alias ExAgent.Message

  @chat_opts_schema [
    temperature: [type: :float, default: 0.6],
    max_tokens: [type: :pos_integer, default: 512],
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
          {:ok, Message.t()} | {:tool_call, String.t(), map()} | {:error, term()}
  def chat(provider, messages, opts \\ []) do
    max_tokens = opts[:max_tokens] || provider.max_tokens
    temperature = opts[:temperature] || provider.temperature

    opts = NimbleOptions.validate!(opts, @chat_opts_schema)

    opts =
      Keyword.merge(opts,
        temperature: temperature,
        max_tokens: max_tokens
      )

    dbg(opts)

    body = build_chat_body(provider.model, messages, provider.tools, provider.system_prompt, opts)

    dbg(body)

    case Req.post(provider.req,
           url: "/chat/completions",
           json: body,
           connect_options: [timeout: :timer.minutes(5)],
           receive_timeout: :timer.minutes(5)
         ) do
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
  defp format_message(%Message{role: :user, content: content, attachments: attachments})
       when is_list(attachments) and attachments != [] do
    file_parts =
      Enum.map(attachments, fn %{data: data, mime_type: mime_type} ->
        if String.starts_with?(mime_type, "image/") do
          %{
            "type" => "image_url",
            "image_url" => %{"url" => "data:#{mime_type};base64,#{Base.encode64(data)}"}
          }
        else
          %{
            "type" => "file",
            "file" => %{
              "file_data" => "data:#{mime_type};base64,#{Base.encode64(data)}"
            }
          }
        end
      end)

    %{
      "role" => "user",
      "content" => file_parts ++ [%{"type" => "text", "text" => content}]
    }
  end

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
      :web_search, acc ->
        Map.put(acc, "web_search_options", %{})

      %{web_search: location_opts}, acc ->
        Map.put(acc, "web_search_options", %{"user_location" => location_opts})

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
