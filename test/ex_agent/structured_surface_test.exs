defmodule ExAgent.StructuredSurfaceTest do
  @moduledoc """
  Bucket 3 of structured output: `:schema` on the public surface - through an
  agent, through a role, and through a stream.

  Verified live before being encoded here: OpenAI accepts `tools` and
  `response_format` in the same request and still emits a tool call, so the
  schema rides on every turn rather than needing a second reformatting request.
  """

  use ExUnit.Case, async: false

  alias ExAgent.Providers.OpenAI
  alias ExAgent.Test.Schemas.Invoice
  alias ExAgent.{Chunk, Error, Roles}

  @invoice_json ~s({"total":128.4,"currency":"EUR","issued_on":"2026-03-14","lines":[]})

  setup do
    on_exit(fn -> Application.delete_env(:ex_agent, :roles) end)
    :ok
  end

  defp provider(plug),
    do: %OpenAI{
      api_key: "sk-test",
      model: "gpt-4o",
      base_url: "http://x",
      req: Req.new(plug: plug)
    }

  defp echo(content) do
    test_pid = self()

    fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, Jason.decode!(body)})

      Req.Test.json(conn, %{
        "choices" => [%{"message" => %{"role" => "assistant", "content" => content}}]
      })
    end
  end

  describe "through an agent" do
    test "given :schema on chat/3, then the response carries the struct" do
      {:ok, agent} = ExAgent.start_agent(provider: provider(echo(@invoice_json)))

      assert {:ok, response} = ExAgent.chat(agent, "Extract the invoice", schema: Invoice)
      assert %Invoice{total: 128.4, currency: :EUR} = response.structured

      assert_received {:request, body}
      assert body["response_format"]["json_schema"]["name"] == "Invoice"
    end

    test "given :schema_doc, then it reaches the wire" do
      {:ok, agent} = ExAgent.start_agent(provider: provider(echo(@invoice_json)))

      assert {:ok, _r} =
               ExAgent.chat(agent, "Extract", schema: Invoice, schema_doc: "totals include tax")

      assert_received {:request, body}

      assert body["response_format"]["json_schema"]["schema"]["description"] ==
               "totals include tax"
    end

    test "given a schema, then only the JSON is stored in history" do
      {:ok, agent} = ExAgent.start_agent(provider: provider(echo(@invoice_json)))

      assert {:ok, _r} = ExAgent.chat(agent, "Extract", schema: Invoice)

      messages = ExAgent.get_context(agent).messages
      assert Enum.map(messages, & &1.role) == [:user, :assistant]
      assert List.last(messages).content == @invoice_json
    end

    test "given a schema on one turn only, then the next turn is unconstrained" do
      {:ok, agent} = ExAgent.start_agent(provider: provider(echo(@invoice_json)))

      assert {:ok, _r} = ExAgent.chat(agent, "Extract", schema: Invoice)
      assert_received {:request, first}
      assert Map.has_key?(first, "response_format")

      assert {:ok, response} = ExAgent.chat(agent, "Now summarise it")
      assert_received {:request, second}
      refute Map.has_key?(second, "response_format")
      assert response.structured == nil
    end
  end

  describe "with tools" do
    test "given a schema and a tool, then the tool runs and the final turn is cast" do
      test_pid = self()
      counter = :counters.new(1, [])

      plug = fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        send(test_pid, {:request, Jason.decode!(body)})
        :counters.add(counter, 1, 1)

        message =
          case :counters.get(counter, 1) do
            1 ->
              %{
                "role" => "assistant",
                "content" => nil,
                "tool_calls" => [
                  %{"id" => "c1", "function" => %{"name" => "rate", "arguments" => "{}"}}
                ]
              }

            _final ->
              %{"role" => "assistant", "content" => @invoice_json}
          end

        Req.Test.json(conn, %{"choices" => [%{"message" => message}]})
      end

      {:ok, tool} =
        ExAgent.Tool.new(
          name: "rate",
          description: "current EUR rate",
          parameters: %{"type" => "object", "properties" => %{}},
          function: fn _args -> {:ok, "1.0"} end
        )

      {:ok, agent} = ExAgent.start_agent(provider: provider(plug), tools: [tool])

      assert {:ok, response} = ExAgent.chat(agent, "Extract the invoice", schema: Invoice)
      assert %Invoice{total: 128.4} = response.structured

      # Both turns carried tools *and* the schema; OpenAI permits the combination.
      assert_received {:request, first}
      assert_received {:request, second}
      assert first["tools"] != nil
      assert first["response_format"] != nil
      assert second["response_format"] != nil
    end
  end

  describe "through a role" do
    # A role resolves a provider from config, so no plug can be injected. What
    # matters here is that `:schema` is *forwarded* rather than dropped, and the
    # capability gate refusing it proves the option arrived.
    test "given chat_with/3 with a schema, then the option reaches the provider gate" do
      Application.put_env(:ex_agent, :roles, extract: {ExAgent.Test.MinimalProvider, []})
      Roles.build!()

      assert {:error, %Error{type: :unsupported}} =
               ExAgent.chat_with(:extract, "Extract", schema: Invoice)

      assert {:ok, response} = ExAgent.chat_with(:extract, "Extract")
      assert response.structured == nil
    end
  end

  describe "through a stream" do
    test "given :schema on chat_stream, then the request is constrained" do
      sse =
        Enum.map_join(
          [
            ~s({"choices":[{"delta":{"content":"{\\"total\\":1.0,"}}]}),
            ~s({"choices":[{"delta":{"content":"\\"currency\\":\\"EUR\\","}}]}),
            ~s({"choices":[{"delta":{"content":"\\"issued_on\\":\\"2026-01-01\\",\\"lines\\":[]}"}}]}),
            ~s({"choices":[{"delta":{},"finish_reason":"stop"}]})
          ],
          "",
          &"data: #{&1}\n\n"
        )

      test_pid = self()

      plug = fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        send(test_pid, {:request, Jason.decode!(body)})

        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, sse <> "data: [DONE]\n\n")
      end

      {:ok, agent} = ExAgent.start_agent(provider: provider(plug))

      chunks = agent |> ExAgent.chat_stream("Extract", schema: Invoice) |> Enum.to_list()
      assert Enum.any?(chunks, &match?(%Chunk{type: :text_delta}, &1))

      assert_received {:request, body}
      assert body["response_format"]["json_schema"]["name"] == "Invoice"
    end

    test "given :schema to collect/2, then the finished turn is cast" do
      sse =
        Enum.map_join(
          [
            ~s({"choices":[{"delta":{"content":#{Jason.encode!(@invoice_json)}}}]}),
            ~s({"choices":[{"delta":{},"finish_reason":"stop"}]})
          ],
          "",
          &"data: #{&1}\n\n"
        )

      plug = fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, sse <> "data: [DONE]\n\n")
      end

      {:ok, agent} = ExAgent.start_agent(provider: provider(plug))

      assert {:ok, response} =
               agent
               |> ExAgent.chat_stream("Extract", schema: Invoice)
               |> ExAgent.collect(schema: Invoice)

      assert %Invoice{total: 128.4, currency: :EUR} = response.structured
    end

    test "given collect/1 with no schema, then :structured stays nil" do
      sse = ~s(data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n)

      plug = fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, sse <> "data: [DONE]\n\n")
      end

      {:ok, agent} = ExAgent.start_agent(provider: provider(plug))

      assert {:ok, response} = agent |> ExAgent.chat_stream("hi") |> ExAgent.collect()
      assert response.structured == nil
    end
  end

  describe "failures" do
    test "given a provider without structured output, then the agent reports it" do
      {:ok, agent} = ExAgent.start_agent(provider: %ExAgent.Test.MinimalProvider{})

      assert {:error, %Error{type: :unsupported}} =
               ExAgent.chat(agent, "Extract", schema: Invoice)
    end

    test "given content that does not fit, then the turn is rolled back" do
      {:ok, agent} = ExAgent.start_agent(provider: provider(echo("not json at all")))

      assert {:error, %Error{type: :invalid_response}} =
               ExAgent.chat(agent, "Extract", schema: Invoice)

      assert ExAgent.get_context(agent).messages == []
    end
  end
end
