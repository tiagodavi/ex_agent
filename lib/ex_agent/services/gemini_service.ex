defmodule ExAgent.Services.GeminiService do
  @moduledoc """
  HTTP service for Google Gemini API.

  Handles payload formatting and response parsing for the
  Gemini `generateContent` endpoint using Req.
  """

  alias ExAgent.{FileRef, Message}
  alias ExAgent.Providers.Gemini

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
          {:ok, Message.t()} | {:tool_call, String.t(), map()} | {:error, term()}
  def chat(%Gemini{} = provider, messages, opts \\ []) do
    opts =
      opts
      |> Keyword.take(Keyword.keys(@chat_opts_schema))
      |> NimbleOptions.validate!(@chat_opts_schema)

    body = build_chat_body(messages, provider.tools, provider.system_prompt, opts)

    case Req.post(provider.req, url: "/models/#{provider.model}:generateContent", json: body) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        parse_response(body)

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, {status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

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
  defp format_attachment(%{file_ref: %FileRef{provider: :gemini, file_uri: uri, mime_type: mt}}) do
    %{"file_data" => %{"file_uri" => uri, "mime_type" => mt}}
  end

  defp format_attachment(%{data: data, mime_type: mime_type}) do
    %{"inline_data" => %{"mime_type" => mime_type, "data" => Base.encode64(data)}}
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
          {:ok, Message.t()} | {:tool_call, String.t(), map()} | {:error, term()}
  defp parse_response(%{"candidates" => [%{"content" => %{"parts" => parts}} | _]}) do
    case parts do
      [%{"functionCall" => %{"name" => name, "args" => args}} | _] ->
        {:tool_call, name, args}

      [%{"text" => text} | _] ->
        {:ok, %Message{role: :assistant, content: text}}

      _ ->
        {:error, {:unexpected_parts, parts}}
    end
  end

  defp parse_response(body), do: {:error, {:unexpected_response, body}}
end
