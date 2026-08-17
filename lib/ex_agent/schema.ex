defmodule ExAgent.Schema do
  @moduledoc """
  Turns an `Ecto` embedded schema into a JSON Schema, and provider JSON back into
  a struct.

  There is no ExAgent-specific schema language. You describe the shape you want
  with an ordinary `Ecto.Schema` and pass the module:

      defmodule Invoice do
        use Ecto.Schema

        @primary_key false
        embedded_schema do
          field :total, :float
          field :currency, Ecto.Enum, values: [:EUR, :USD, :GBP]
          embeds_many :lines, Invoice.Line
        end
      end

      {:ok, response} = ExAgent.chat(agent, "Extract the invoice", schema: Invoice)
      response.structured   #=> %Invoice{total: 128.4, currency: :EUR, lines: [...]}

  `Ecto` is an optional dependency of ExAgent, needed only for this.

  ## The rules, and where they come from

  Each was verified against a live API rather than inferred:

  - **Every field is required.** OpenAI's strict mode rejects a schema whose
    `required` list omits any property, and Ecto has no field-level `required`
    anyway, so the two agree. Optional fields cannot be expressed.
  - **A root list is wrapped.** OpenAI refuses a schema whose root is
    `type: "array"`, so `{:list, Invoice}` becomes an object with a single
    `"items"` property, unwrapped again by `cast/2`. Callers never see it.
  - **`additionalProperties: false` is required by OpenAI and rejected by
    Gemini**, hence `:dialect`.
  - **Primary keys are excluded.** `embedded_schema` carries a `binary_id`
    primary key by default, and a primary key is identity the application
    assigns, not a fact a model extracts. Asking for one invites a plausible
    hallucinated id.
  """

  alias ExAgent.Error

  @type schema :: module() | {:list, module()}
  @type dialect :: :openai | :gemini

  @dialects [:openai, :gemini]

  @doc """
  Builds a JSON Schema from an Ecto embedded schema.

  ## Options

  - `:description` - becomes the schema root's `"description"`, which both
    OpenAI and Gemini pass to the model
  - `:dialect` - `:openai` (default) emits `additionalProperties: false`;
    `:gemini` omits it, because Gemini rejects the key outright

  ## Examples

      {:ok, json} = ExAgent.Schema.to_json_schema(Invoice, description: "One invoice")

      json["required"]              #=> ["total", "currency", "issued_on", "lines"]
      json["additionalProperties"]  #=> false
  """
  @spec to_json_schema(schema(), keyword()) :: {:ok, map()} | {:error, Error.t()}
  def to_json_schema(schema, opts \\ []) do
    dialect = Keyword.get(opts, :dialect, :openai)

    with :ok <- validate_dialect(dialect),
         {:ok, json} <- build(schema, dialect) do
      {:ok, maybe_describe(json, opts[:description])}
    end
  end

  @doc """
  Casts decoded provider JSON into the schema's struct.

  Returns `{:error, %ExAgent.Error{type: :invalid_response}}` when the JSON does
  not fit: a wrong type, a value outside an `Ecto.Enum`, or a missing field. A
  missing field is an error rather than a `nil` because an endpoint that ignored
  the schema would otherwise hand back a half-built struct with no signal.
  """
  @spec cast(schema(), term()) :: {:ok, struct() | [struct()]} | {:error, Error.t()}
  def cast({:list, module}, %{"items" => items}) when is_list(items) do
    items
    |> Enum.reduce_while({:ok, []}, fn item, {:ok, acc} ->
      case cast(module, item) do
        {:ok, struct} -> {:cont, {:ok, [struct | acc]}}
        {:error, _reason} = failure -> {:halt, failure}
      end
    end)
    |> case do
      {:ok, structs} -> {:ok, Enum.reverse(structs)}
      {:error, _reason} = failure -> failure
    end
  end

  def cast({:list, _module}, params) do
    {:error,
     invalid_response("expected an object with an \"items\" array, got #{inspect(params)}")}
  end

  def cast(module, params) when is_atom(module) and is_map(params) do
    with :ok <- ensure_schema(module),
         :ok <- check_present(module, params) do
      module
      |> changeset(struct(module), params)
      |> Ecto.Changeset.apply_action(:insert)
      |> case do
        {:ok, _struct} = success -> success
        {:error, changeset} -> {:error, invalid_response(describe_errors(changeset))}
      end
    end
  end

  def cast(_module, params),
    do: {:error, invalid_response("expected a JSON object, got #{inspect(params)}")}

  # --- JSON Schema construction ---

  @spec build(schema(), dialect()) :: {:ok, map()} | {:error, Error.t()}
  defp build({:list, module}, dialect) do
    with {:ok, item} <- build(module, dialect) do
      {:ok, object(%{"items" => %{"type" => "array", "items" => item}}, ["items"], dialect)}
    end
  end

  defp build(module, dialect) when is_atom(module) do
    with :ok <- ensure_schema(module),
         {:ok, fields} <- schema_fields(module),
         {:ok, properties} <- properties(module, fields, dialect) do
      {:ok, object(properties, Enum.map(fields, &to_string/1), dialect)}
    end
  end

  # `:schema` is deliberately loosely typed at the call site because this module
  # owns the vocabulary, which means the wrong shape arrives here rather than
  # being caught by NimbleOptions. A raw JSON Schema map and `[Invoice]` are the
  # two likely mistakes, so both are named.
  defp build(other, _dialect) do
    {:error,
     invalid_request(
       ":schema must be an Ecto schema module or {:list, module}, got #{inspect(other)}" <>
         list_hint(other)
     )}
  end

  @spec list_hint(term()) :: String.t()
  defp list_hint([module]) when is_atom(module),
    do: "; for a list of results use {:list, #{inspect(module)}}"

  defp list_hint(map) when is_map(map),
    do: "; a raw JSON Schema map is not accepted here, only for a tool's :parameters"

  defp list_hint(_other), do: ""

  @spec object(map(), [String.t()], dialect()) :: map()
  defp object(properties, required, dialect) do
    base = %{"type" => "object", "properties" => properties, "required" => required}

    # Gemini's responseSchema has no `additionalProperties` field and 400s on it;
    # OpenAI's strict mode requires it to be present and false.
    if dialect == :openai, do: Map.put(base, "additionalProperties", false), else: base
  end

  @spec properties(module(), [atom()], dialect()) :: {:ok, map()} | {:error, Error.t()}
  defp properties(module, fields, dialect) do
    Enum.reduce_while(fields, {:ok, %{}}, fn field, {:ok, acc} ->
      case property(module, field, dialect) do
        {:ok, json} -> {:cont, {:ok, Map.put(acc, to_string(field), json)}}
        {:error, _reason} = failure -> {:halt, failure}
      end
    end)
  end

  @spec property(module(), atom(), dialect()) :: {:ok, map()} | {:error, Error.t()}
  defp property(module, field, dialect) do
    case module.__schema__(:embed, field) do
      nil -> scalar(module, field)
      embed -> embedded(embed, dialect)
    end
  end

  @spec embedded(Ecto.Embedded.t(), dialect()) :: {:ok, map()} | {:error, Error.t()}
  defp embedded(%Ecto.Embedded{cardinality: :one, related: related}, dialect),
    do: build(related, dialect)

  defp embedded(%Ecto.Embedded{cardinality: :many, related: related}, dialect) do
    with {:ok, item} <- build(related, dialect) do
      {:ok, %{"type" => "array", "items" => item}}
    end
  end

  @spec scalar(module(), atom()) :: {:ok, map()} | {:error, Error.t()}
  defp scalar(module, field) do
    case json_type(module.__schema__(:type, field), module, field) do
      {:ok, _json} = success -> success
      :error -> {:error, unsupported_type(module, field)}
    end
  end

  # Ecto.Enum is a parameterized type whose tuple shape has changed across Ecto
  # versions, so it is detected through the public `Ecto.Enum.values/2` rather
  # than by matching the tuple.
  @spec json_type(term(), module(), atom()) :: {:ok, map()} | :error
  defp json_type({:parameterized, _params}, module, field), do: enum_type(module, field)
  defp json_type({:parameterized, _mod, _params}, module, field), do: enum_type(module, field)

  defp json_type({:array, inner}, module, field) do
    case json_type(inner, module, field) do
      {:ok, items} -> {:ok, %{"type" => "array", "items" => items}}
      :error -> :error
    end
  end

  defp json_type(:string, _module, _field), do: {:ok, %{"type" => "string"}}
  defp json_type(:binary_id, _module, _field), do: {:ok, %{"type" => "string"}}
  defp json_type(:integer, _module, _field), do: {:ok, %{"type" => "integer"}}
  defp json_type(:id, _module, _field), do: {:ok, %{"type" => "integer"}}
  defp json_type(:float, _module, _field), do: {:ok, %{"type" => "number"}}
  defp json_type(:decimal, _module, _field), do: {:ok, %{"type" => "number"}}
  defp json_type(:boolean, _module, _field), do: {:ok, %{"type" => "boolean"}}
  defp json_type(:date, _module, _field), do: {:ok, string_format("date")}
  defp json_type(:time, _module, _field), do: {:ok, string_format("time")}
  defp json_type(:utc_datetime, _module, _field), do: {:ok, string_format("date-time")}
  defp json_type(:naive_datetime, _module, _field), do: {:ok, string_format("date-time")}
  defp json_type(:utc_datetime_usec, _module, _field), do: {:ok, string_format("date-time")}
  defp json_type(:naive_datetime_usec, _module, _field), do: {:ok, string_format("date-time")}
  defp json_type(Ecto.UUID, _module, _field), do: {:ok, %{"type" => "string"}}
  defp json_type(_type, _module, _field), do: :error

  @spec enum_type(module(), atom()) :: {:ok, map()} | :error
  defp enum_type(module, field) do
    values = module |> Ecto.Enum.values(field) |> Enum.map(&to_string/1)
    {:ok, %{"type" => "string", "enum" => values}}
  rescue
    # A parameterized type that is not an Ecto.Enum, which has no general JSON
    # Schema equivalent.
    _error -> :error
  end

  @spec string_format(String.t()) :: map()
  defp string_format(format), do: %{"type" => "string", "format" => format}

  # --- Casting ---

  @spec changeset(module(), struct(), map()) :: Ecto.Changeset.t()
  defp changeset(module, data, params) do
    embeds = module.__schema__(:embeds)
    scalars = module.__schema__(:fields) -- embeds

    # `empty_values: []` keeps `""` as `""`. Ecto's default turns it into nil,
    # which would erase the difference between a model that found nothing and a
    # field it never returned - and the latter is already an error above.
    cast =
      Ecto.Changeset.cast(data, params, drop_primary_key(module, scalars), empty_values: [])

    Enum.reduce(embeds, cast, fn
      field, acc ->
        related = module.__schema__(:embed, field).related

        Ecto.Changeset.cast_embed(acc, field,
          with: fn embedded_data, embedded_params ->
            changeset(related, embedded_data, embedded_params)
          end
        )
    end)
  end

  # A field the model was never asked for cannot be cast, and leaving it nil would
  # hide an endpoint that ignored the schema entirely.
  @spec check_present(module(), map()) :: :ok | {:error, Error.t()}
  defp check_present(module, params) do
    case Enum.reject(expected_fields(module), &Map.has_key?(params, to_string(&1))) do
      [] ->
        :ok

      missing ->
        {:error,
         invalid_response(
           "missing #{Enum.map_join(missing, ", ", &inspect/1)} for " <>
             "#{inspect(module)}; the model returned #{inspect(Map.keys(params))}"
         )}
    end
  end

  @spec describe_errors(Ecto.Changeset.t()) :: String.t()
  defp describe_errors(changeset) do
    changeset
    |> Ecto.Changeset.traverse_errors(fn {message, _opts} -> message end)
    |> flatten_errors()
    |> Enum.map_join("; ", fn {field, message} -> "#{field} #{message}" end)
  end

  @spec flatten_errors(map(), String.t() | nil) :: [{String.t(), String.t()}]
  defp flatten_errors(errors, prefix \\ nil) do
    Enum.flat_map(errors, fn {field, value} ->
      name = if prefix, do: "#{prefix}.#{field}", else: to_string(field)
      flatten_error(name, value)
    end)
  end

  @spec flatten_error(String.t(), term()) :: [{String.t(), String.t()}]
  defp flatten_error(name, messages) when is_list(messages) do
    Enum.flat_map(messages, fn
      message when is_binary(message) -> [{name, message}]
      nested when is_map(nested) -> flatten_errors(nested, name)
      _other -> []
    end)
  end

  defp flatten_error(name, nested) when is_map(nested), do: flatten_errors(nested, name)
  defp flatten_error(_name, _other), do: []

  # --- Reflection helpers ---

  @spec schema_fields(module()) :: {:ok, [atom()]} | {:error, Error.t()}
  defp schema_fields(module) do
    case expected_fields(module) do
      [] ->
        {:error,
         invalid_request(
           "#{inspect(module)} has no fields to extract; add at least one field " <>
             "to its embedded_schema"
         )}

      fields ->
        {:ok, fields}
    end
  end

  @spec expected_fields(module()) :: [atom()]
  defp expected_fields(module),
    do: drop_primary_key(module, module.__schema__(:fields))

  @spec drop_primary_key(module(), [atom()]) :: [atom()]
  defp drop_primary_key(module, fields), do: fields -- module.__schema__(:primary_key)

  @spec ensure_schema(module()) :: :ok | {:error, Error.t()}
  defp ensure_schema(module) do
    if Code.ensure_loaded?(module) and function_exported?(module, :__schema__, 1) do
      :ok
    else
      {:error,
       invalid_request(
         "#{inspect(module)} is not an Ecto schema; :schema takes a module using " <>
           "Ecto.Schema with an embedded_schema block"
       )}
    end
  end

  @spec validate_dialect(term()) :: :ok | {:error, Error.t()}
  defp validate_dialect(dialect) when dialect in @dialects, do: :ok

  defp validate_dialect(dialect),
    do:
      {:error,
       invalid_request(
         "unknown dialect #{inspect(dialect)}, expected one of #{inspect(@dialects)}"
       )}

  @spec maybe_describe(map(), String.t() | nil) :: map()
  defp maybe_describe(json, nil), do: json
  defp maybe_describe(json, ""), do: json
  defp maybe_describe(json, description), do: Map.put(json, "description", description)

  @spec unsupported_type(module(), atom()) :: Error.t()
  defp unsupported_type(module, field) do
    invalid_request(
      "#{inspect(module)} field #{inspect(field)} has type " <>
        "#{inspect(module.__schema__(:type, field))}, which has no JSON Schema " <>
        "equivalent; use a typed field or an embedded schema instead"
    )
  end

  @spec invalid_request(String.t()) :: Error.t()
  defp invalid_request(message), do: Error.new(:invalid_request, message, __MODULE__)

  @spec invalid_response(String.t()) :: Error.t()
  defp invalid_response(message),
    do:
      Error.new(
        :invalid_response,
        "model returned JSON that does not fit: " <> message,
        __MODULE__
      )
end
