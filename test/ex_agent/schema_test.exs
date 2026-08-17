defmodule ExAgent.SchemaTest do
  @moduledoc """
  Bucket 1 of structured output: Ecto reflection to JSON Schema, and JSON back to
  a struct. No providers, no network.

  The rules asserted here were verified against live APIs before being encoded:
  OpenAI rejects an array root, rejects a property missing from `required`, and
  rejects a schema without `additionalProperties: false`; Gemini rejects
  `additionalProperties` outright.
  """

  use ExUnit.Case, async: true

  alias ExAgent.{Error, Schema}
  alias ExAgent.Test.Schemas.{Empty, FreeMap, Invoice, Line, Person, WithPrimaryKey}

  describe "to_json_schema/2 happy paths" do
    test "given an embedded schema, then every field is a property and all are required" do
      assert {:ok, json} = Schema.to_json_schema(Invoice)

      assert json["type"] == "object"
      assert json["additionalProperties"] == false

      assert Enum.sort(Map.keys(json["properties"])) == [
               "currency",
               "issued_on",
               "lines",
               "total"
             ]

      assert Enum.sort(json["required"]) == ["currency", "issued_on", "lines", "total"]
    end

    test "given Ecto types, then each maps to its JSON Schema counterpart" do
      assert {:ok, json} = Schema.to_json_schema(Person)
      props = json["properties"]

      assert props["name"] == %{"type" => "string"}
      assert props["age"] == %{"type" => "integer"}
      assert props["active"] == %{"type" => "boolean"}
      assert props["tags"] == %{"type" => "array", "items" => %{"type" => "string"}}
      assert props["seen_at"] == %{"type" => "string", "format" => "date-time"}
    end

    test "given nested embeds, then the inner shape is described inline" do
      assert {:ok, json} = Schema.to_json_schema(Invoice)

      lines = json["properties"]["lines"]
      assert lines["type"] == "array"
      assert lines["items"]["type"] == "object"
      assert Enum.sort(Map.keys(lines["items"]["properties"])) == ["amount", "description"]
      assert lines["items"]["additionalProperties"] == false
    end

    test "given an Ecto.Enum, then its values become a JSON Schema enum" do
      assert {:ok, json} = Schema.to_json_schema(Invoice)

      assert json["properties"]["currency"] == %{
               "type" => "string",
               "enum" => ["EUR", "USD", "GBP"]
             }
    end

    test "given a :description, then it lands on the schema root" do
      assert {:ok, json} = Schema.to_json_schema(Invoice, description: "One supplier invoice")
      assert json["description"] == "One supplier invoice"
    end

    test "given {:list, schema}, then the root is an object wrapping an items array" do
      assert {:ok, json} = Schema.to_json_schema({:list, Invoice})

      assert json["type"] == "object"
      assert json["required"] == ["items"]
      assert json["properties"]["items"]["type"] == "array"
      assert json["properties"]["items"]["items"]["type"] == "object"
    end

    test "given the :gemini dialect, then additionalProperties is stripped everywhere" do
      assert {:ok, json} = Schema.to_json_schema(Invoice, dialect: :gemini)

      refute Map.has_key?(json, "additionalProperties")
      refute Map.has_key?(json["properties"]["lines"]["items"], "additionalProperties")
      assert json["properties"]["lines"]["items"]["type"] == "object"
    end
  end

  describe "to_json_schema/2 failure paths" do
    test "given a module that is not an Ecto schema, then it is refused by name" do
      assert {:error, %Error{type: :invalid_request} = error} = Schema.to_json_schema(URI)
      assert error.message =~ "URI is not an Ecto schema"
    end

    test "given a field type with no JSON Schema equivalent, then the field is named" do
      assert {:error, %Error{type: :invalid_request} = error} = Schema.to_json_schema(FreeMap)
      assert error.message =~ ":payload"
      assert error.message =~ ":map"
    end

    test "given a schema with no fields, then it is refused rather than sent empty" do
      assert {:error, %Error{type: :invalid_request} = error} = Schema.to_json_schema(Empty)
      assert error.message =~ "no fields"
    end

    test "given an unknown dialect, then it is refused" do
      assert {:error, %Error{type: :invalid_request}} =
               Schema.to_json_schema(Invoice, dialect: :nope)
    end
  end

  describe "to_json_schema/2 edge cases" do
    test "given a schema keeping Ecto's default primary key, then :id is excluded" do
      assert {:ok, json} = Schema.to_json_schema(WithPrimaryKey)

      assert Map.keys(json["properties"]) == ["name"]
      assert json["required"] == ["name"]
    end

    test "given a :date field, then it is a string with a date format" do
      assert {:ok, json} = Schema.to_json_schema(Invoice)
      assert json["properties"]["issued_on"] == %{"type" => "string", "format" => "date"}
    end

    test "given {:list, schema} with a description, then it describes the wrapper root" do
      assert {:ok, json} = Schema.to_json_schema({:list, Invoice}, description: "Every invoice")
      assert json["description"] == "Every invoice"
    end
  end

  describe "cast/2 happy paths" do
    test "given conforming JSON, then a struct comes back with types cast" do
      params = %{
        "total" => 128.4,
        "currency" => "EUR",
        "issued_on" => "2026-03-14",
        "lines" => [%{"description" => "Consulting", "amount" => 128.4}]
      }

      assert {:ok, %Invoice{} = invoice} = Schema.cast(Invoice, params)
      assert invoice.total == 128.4
      assert invoice.currency == :EUR
      assert invoice.issued_on == ~D[2026-03-14]
      assert [%Line{description: "Consulting", amount: 128.4}] = invoice.lines
    end

    test "given a wrapped list, then a list of structs comes back" do
      params = %{
        "items" => [
          %{"total" => 1.0, "currency" => "EUR", "issued_on" => "2026-01-01", "lines" => []},
          %{"total" => 2.0, "currency" => "USD", "issued_on" => "2026-01-02", "lines" => []}
        ]
      }

      assert {:ok, [%Invoice{total: 1.0}, %Invoice{total: 2.0, currency: :USD}]} =
               Schema.cast({:list, Invoice}, params)
    end

    test "given a nested embeds_one, then the inner struct is built" do
      params = %{
        "name" => "Ada",
        "age" => 36,
        "active" => true,
        "tags" => ["math"],
        "seen_at" => "2026-03-14T10:00:00Z",
        "address" => %{"city" => "London"}
      }

      assert {:ok, %Person{} = person} = Schema.cast(Person, params)
      assert person.address.city == "London"
      assert person.seen_at == ~U[2026-03-14 10:00:00Z]
    end
  end

  describe "cast/2 failure paths" do
    test "given a value of the wrong type, then the error names the field" do
      params = %{
        "total" => "not a number",
        "currency" => "EUR",
        "issued_on" => "2026-03-14",
        "lines" => []
      }

      assert {:error, %Error{type: :invalid_response} = error} = Schema.cast(Invoice, params)
      assert error.message =~ "total"
    end

    test "given a value outside an Ecto.Enum, then the error names the field" do
      params = %{"total" => 1.0, "currency" => "BRL", "issued_on" => "2026-03-14", "lines" => []}

      assert {:error, %Error{type: :invalid_response} = error} = Schema.cast(Invoice, params)
      assert error.message =~ "currency"
    end

    test "given a missing field, then it is reported rather than left nil" do
      assert {:error, %Error{type: :invalid_response} = error} =
               Schema.cast(Invoice, %{"total" => 1.0})

      assert error.message =~ "currency"
      assert error.message =~ "issued_on"
    end

    test "given a wrapped list whose items key is absent, then it is reported" do
      assert {:error, %Error{type: :invalid_response}} = Schema.cast({:list, Invoice}, %{})
    end
  end

  describe "cast/2 edge cases" do
    test "given an empty string for a string field, then it is accepted" do
      params = %{
        "name" => "",
        "age" => 0,
        "active" => false,
        "tags" => [],
        "seen_at" => "2026-03-14T10:00:00Z",
        "address" => %{"city" => ""}
      }

      assert {:ok, %Person{name: "", tags: [], active: false}} = Schema.cast(Person, params)
    end

    test "given an empty items list, then an empty list comes back" do
      assert {:ok, []} = Schema.cast({:list, Invoice}, %{"items" => []})
    end

    test "given a schema with a primary key, then the id stays nil and is not required" do
      assert {:ok, %WithPrimaryKey{id: nil, name: "x"}} =
               Schema.cast(WithPrimaryKey, %{"name" => "x"})
    end

    test "given nested embeds with a bad inner value, then the inner field is named" do
      params = %{
        "total" => 1.0,
        "currency" => "EUR",
        "issued_on" => "2026-03-14",
        "lines" => [%{"description" => "ok", "amount" => "not a float"}]
      }

      assert {:error, %Error{type: :invalid_response} = error} = Schema.cast(Invoice, params)
      assert error.message =~ "amount"
    end

    test "given params that are not a map, then it is refused" do
      assert {:error, %Error{type: :invalid_response}} = Schema.cast(Invoice, "nope")
    end
  end
end
