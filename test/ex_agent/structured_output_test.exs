defmodule ExAgent.StructuredOutputTest do
  @moduledoc """
  Bucket 2 of structured output: what each provider puts on the wire for a
  `:schema`, and what comes back.

  The per-provider differences asserted here were verified against live APIs:
  OpenAI requires `additionalProperties: false` and an object root, Gemini 400s
  on `additionalProperties` and wants `responseMimeType` alongside
  `responseSchema`, and vLLM honours `response_format` but silently ignores
  `guided_json`.
  """

  use ExUnit.Case, async: true

  alias ExAgent.Providers.{Gemini, OpenAI, OpenAICompatible}
  alias ExAgent.Test.Schemas.{Invoice, Line}
  alias ExAgent.{Error, Message, Provider, Response}

  @invoice_json ~s({"total":128.4,"currency":"EUR","issued_on":"2026-03-14",) <>
                  ~s("lines":[{"description":"Consulting","amount":128.4}]})

  defp user_msg(content \\ "Extract the invoice") do
    {:ok, message} = Message.new(role: :user, content: content)
    message
  end

  # Records the request body, answers with `content` as the assistant message.
  defp openai_style(module, content, extra \\ %{}) do
    test_pid = self()

    plug = fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, Jason.decode!(body)})

      message = Map.merge(%{"role" => "assistant", "content" => content}, extra)
      Req.Test.json(conn, %{"choices" => [%{"message" => message}]})
    end

    base =
      case module do
        OpenAI ->
          %OpenAI{api_key: "sk-test", model: "gpt-4o", base_url: "http://x"}

        OpenAICompatible ->
          %OpenAICompatible{model: "Qwen/Qwen3-VL", base_url: "http://x", modalities: [:text]}
      end

    struct!(base, req: Req.new(plug: plug))
  end

  defp gemini_style(text) do
    test_pid = self()

    plug = fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, Jason.decode!(body)})

      Req.Test.json(conn, %{
        "candidates" => [%{"content" => %{"parts" => [%{"text" => text}]}}]
      })
    end

    %Gemini{
      api_key: "k",
      model: "gemini-3.6-flash",
      base_url: "http://x",
      req: Req.new(plug: plug)
    }
  end

  describe "OpenAI request shaping" do
    test "given a schema, then response_format carries a strict json_schema" do
      provider = openai_style(OpenAI, @invoice_json)

      assert {:ok, _response} = Provider.chat(provider, [user_msg()], schema: Invoice)

      assert_received {:request, body}
      format = body["response_format"]

      assert format["type"] == "json_schema"
      assert format["json_schema"]["strict"] == true
      assert format["json_schema"]["name"] == "Invoice"

      schema = format["json_schema"]["schema"]
      assert schema["type"] == "object"
      assert schema["additionalProperties"] == false
      assert Enum.sort(schema["required"]) == ["currency", "issued_on", "lines", "total"]
    end

    test "given :schema_doc, then it becomes the schema root description" do
      provider = openai_style(OpenAI, @invoice_json)

      assert {:ok, _response} =
               Provider.chat(provider, [user_msg()],
                 schema: Invoice,
                 schema_doc: "issued_on is the date on the invoice, not today"
               )

      assert_received {:request, body}

      assert body["response_format"]["json_schema"]["schema"]["description"] =~
               "not today"
    end

    test "given no schema, then response_format is absent" do
      provider = openai_style(OpenAI, "plain text")

      assert {:ok, _response} = Provider.chat(provider, [user_msg()])

      assert_received {:request, body}
      refute Map.has_key?(body, "response_format")
    end

    test "given {:list, schema}, then the wire schema wraps an items array" do
      provider = openai_style(OpenAI, ~s({"items":[]}))

      assert {:ok, _response} = Provider.chat(provider, [user_msg()], schema: {:list, Invoice})

      assert_received {:request, body}
      schema = body["response_format"]["json_schema"]["schema"]

      assert schema["type"] == "object"
      assert schema["required"] == ["items"]
      assert schema["properties"]["items"]["type"] == "array"
    end
  end

  describe "OpenAICompatible request shaping" do
    test "given a schema, then response_format is used rather than guided_json" do
      provider = openai_style(OpenAICompatible, @invoice_json)

      assert {:ok, _response} = Provider.chat(provider, [user_msg()], schema: Invoice)

      assert_received {:request, body}

      assert body["response_format"]["type"] == "json_schema"
      # vLLM accepts `guided_json` and then ignores it, returning prose.
      refute Map.has_key?(body, "guided_json")
    end
  end

  describe "Gemini request shaping" do
    test "given a schema, then responseSchema and responseMimeType are set" do
      provider = gemini_style(@invoice_json)

      assert {:ok, _response} = Provider.chat(provider, [user_msg()], schema: Invoice)

      assert_received {:request, body}
      config = body["generationConfig"]

      assert config["responseMimeType"] == "application/json"
      assert config["responseSchema"]["type"] == "object"
    end

    test "given a schema, then additionalProperties is absent at every level" do
      provider = gemini_style(@invoice_json)

      assert {:ok, _response} = Provider.chat(provider, [user_msg()], schema: Invoice)

      assert_received {:request, body}
      schema = body["generationConfig"]["responseSchema"]

      refute Map.has_key?(schema, "additionalProperties")
      refute Map.has_key?(schema["properties"]["lines"]["items"], "additionalProperties")
    end
  end

  describe "casting the response" do
    test "given conforming JSON, then :structured holds the struct" do
      provider = openai_style(OpenAI, @invoice_json)

      assert {:ok, %Response{} = response} =
               Provider.chat(provider, [user_msg()], schema: Invoice)

      assert %Invoice{total: 128.4, currency: :EUR, issued_on: ~D[2026-03-14]} =
               response.structured

      assert [%Line{description: "Consulting"}] = response.structured.lines
      assert response.content == @invoice_json
    end

    test "given a wrapped list, then :structured holds a plain list" do
      body = ~s({"items":[{"total":1.0,"currency":"EUR","issued_on":"2026-01-01","lines":[]}]})
      provider = openai_style(OpenAI, body)

      assert {:ok, response} = Provider.chat(provider, [user_msg()], schema: {:list, Invoice})
      assert [%Invoice{total: 1.0}] = response.structured
    end

    test "given Gemini JSON, then :structured holds the struct" do
      provider = gemini_style(@invoice_json)

      assert {:ok, response} = Provider.chat(provider, [user_msg()], schema: Invoice)
      assert %Invoice{currency: :EUR} = response.structured
    end

    test "given no schema, then :structured stays nil" do
      provider = openai_style(OpenAI, "plain text")

      assert {:ok, response} = Provider.chat(provider, [user_msg()])
      assert response.structured == nil
    end
  end

  describe "failures" do
    test "given a schema that cannot be built, then no request is made" do
      provider = openai_style(OpenAI, @invoice_json)

      assert {:error, %Error{type: :invalid_request}} =
               Provider.chat(provider, [user_msg()], schema: URI)

      refute_received {:request, _body}
    end

    test "given content that is not JSON, then the failure names the provider" do
      provider = openai_style(OpenAI, "I'm afraid I can't do that.")

      assert {:error, %Error{type: :invalid_response} = error} =
               Provider.chat(provider, [user_msg()], schema: Invoice)

      assert error.message =~ "not valid JSON"
    end

    test "given JSON that does not fit the schema, then the field is named" do
      provider =
        openai_style(
          OpenAI,
          ~s({"total":"lots","currency":"EUR","issued_on":"2026-03-14","lines":[]})
        )

      assert {:error, %Error{type: :invalid_response} = error} =
               Provider.chat(provider, [user_msg()], schema: Invoice)

      assert error.message =~ "total"
    end

    test "given an OpenAI refusal, then it is reported as a refusal" do
      provider = openai_style(OpenAI, nil, %{"refusal" => "I cannot extract that."})

      assert {:error, %Error{type: :refusal} = error} =
               Provider.chat(provider, [user_msg()], schema: Invoice)

      assert error.message =~ "cannot extract"
    end
  end

  describe "edge cases" do
    test "given a schema and tools, then a tool call still short-circuits" do
      test_pid = self()

      plug = fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        send(test_pid, {:request, Jason.decode!(body)})

        Req.Test.json(conn, %{
          "choices" => [
            %{
              "message" => %{
                "role" => "assistant",
                "content" => nil,
                "tool_calls" => [
                  %{"id" => "c1", "function" => %{"name" => "lookup", "arguments" => "{}"}}
                ]
              }
            }
          ]
        })
      end

      {:ok, tool} =
        ExAgent.Tool.new(
          name: "lookup",
          description: "look up",
          parameters: %{"type" => "object", "properties" => %{}},
          function: fn _args -> {:ok, "x"} end
        )

      provider = %OpenAI{
        api_key: "sk-test",
        model: "gpt-4o",
        base_url: "http://x",
        tools: [tool],
        req: Req.new(plug: plug)
      }

      assert {:tool_calls, [%{"name" => "lookup"}]} =
               Provider.chat(provider, [user_msg()], schema: Invoice)

      assert_received {:request, body}
      assert body["response_format"]["type"] == "json_schema"
    end

    test "given empty JSON content, then it is reported rather than cast to nil" do
      provider = openai_style(OpenAI, "")

      assert {:error, %Error{type: :invalid_response}} =
               Provider.chat(provider, [user_msg()], schema: Invoice)
    end

    test "given a schema on a provider without structured output, then it is refused" do
      provider = %ExAgent.Test.MinimalProvider{}

      assert {:error, %Error{type: :unsupported}} =
               Provider.chat(provider, [user_msg()], schema: Invoice)
    end
  end
end
