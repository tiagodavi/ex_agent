defmodule ExAgent.Services.Structured do
  @moduledoc false

  # Shared plumbing for the `:schema` option. The *vocabularies* differ per
  # provider and stay in each service: OpenAI wants a `response_format` with
  # `strict: true` and `additionalProperties: false`, Gemini wants
  # `responseSchema` plus `responseMimeType` and rejects `additionalProperties`
  # outright. What is identical is building the JSON Schema and casting the
  # answer back, so only that lives here.

  alias ExAgent.{Error, Response, Schema}

  @doc false
  @spec json_schema(Schema.schema() | nil, Schema.dialect(), keyword()) ::
          {:ok, map() | nil} | {:error, Error.t()}
  def json_schema(nil, _dialect, _opts), do: {:ok, nil}

  def json_schema(schema, dialect, opts) do
    Schema.to_json_schema(schema, dialect: dialect, description: opts[:schema_doc])
  end

  @doc false
  # OpenAI requires a name matching ~r/^[a-zA-Z0-9_-]+$/, so the module's last
  # segment is used rather than its full name.
  @spec name(Schema.schema()) :: String.t()
  def name({:list, module}), do: name(module) <> "List"
  def name(module), do: module |> Module.split() |> List.last()

  @doc false
  # Wraps a service's parsed result, casting the assistant text into the schema's
  # struct. A `{:tool_calls, _}` result passes through untouched: the tool loop
  # has not reached the final turn yet, so there is nothing to cast.
  @spec decode(term(), Schema.schema() | nil, module()) :: term()
  def decode(result, nil, _provider), do: result

  def decode({:ok, %Response{content: content} = response}, schema, provider) do
    with {:ok, decoded} <- decode_json(content, provider),
         {:ok, structured} <- cast(schema, decoded, provider) do
      {:ok, %{response | structured: structured}}
    end
  end

  def decode(result, _schema, _provider), do: result

  @spec decode_json(term(), module()) :: {:ok, term()} | {:error, Error.t()}
  defp decode_json(content, provider) when is_binary(content) and content != "" do
    case Jason.decode(content) do
      {:ok, decoded} ->
        {:ok, decoded}

      {:error, _reason} ->
        {:error,
         Error.new(
           :invalid_response,
           "a schema was requested but the model's answer is not valid JSON: " <>
             inspect(String.slice(content, 0, 200)),
           provider
         )}
    end
  end

  defp decode_json(content, provider) do
    {:error,
     Error.new(
       :invalid_response,
       "a schema was requested but the model returned no content to decode, got " <>
         inspect(content),
       provider
     )}
  end

  # `ExAgent.Schema` tags its errors with itself; the caller cares which provider
  # produced the answer that did not fit.
  @spec cast(Schema.schema(), term(), module()) :: {:ok, term()} | {:error, Error.t()}
  defp cast(schema, decoded, provider) do
    case Schema.cast(schema, decoded) do
      {:ok, _structured} = success -> success
      {:error, error} -> {:error, %{error | provider: provider}}
    end
  end
end
