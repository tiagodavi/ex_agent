defmodule ExAgent.TypedToolTest do
  @moduledoc """
  Bucket 4 of structured output: an `Ecto` schema as a tool's `:parameters`, so
  the tool function receives a struct instead of a string-keyed map.
  """

  use ExUnit.Case, async: true

  alias ExAgent.Providers.OpenAI
  alias ExAgent.Test.Schemas.{Address, Person}
  alias ExAgent.Tool

  defmodule Query do
    @moduledoc false
    use Ecto.Schema

    @primary_key false
    embedded_schema do
      field :order_id, :string
    end
  end

  defp provider(plug),
    do: %OpenAI{
      api_key: "sk-test",
      model: "gpt-4o",
      base_url: "http://x",
      req: Req.new(plug: plug)
    }

  # First turn calls the tool with `args`, second turn answers.
  defp tool_calling_plug(args) do
    counter = :counters.new(1, [])

    fn conn ->
      :counters.add(counter, 1, 1)

      message =
        case :counters.get(counter, 1) do
          1 ->
            %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{"id" => "c1", "function" => %{"name" => "lookup", "arguments" => args}}
              ]
            }

          _final ->
            %{"role" => "assistant", "content" => "done"}
        end

      Req.Test.json(conn, %{"choices" => [%{"message" => message}]})
    end
  end

  describe "Tool.new/1 with a schema module" do
    test "given a schema module, then :parameters holds its JSON Schema" do
      assert {:ok, tool} =
               Tool.new(
                 name: "lookup",
                 description: "Look up an order",
                 parameters: Query,
                 function: fn _args -> :ok end
               )

      assert tool.parameters["type"] == "object"
      assert tool.parameters["properties"]["order_id"] == %{"type" => "string"}
      assert tool.parameters["required"] == ["order_id"]
      assert tool.schema == Query
    end

    test "given a schema module, then additionalProperties is omitted" do
      assert {:ok, tool} =
               Tool.new(
                 name: "lookup",
                 description: "d",
                 parameters: Query,
                 function: fn _args -> :ok end
               )

      # Gemini rejects the key in a function declaration, and OpenAI's
      # non-strict function calling does not require it, so the shape both
      # accept is the one without it.
      refute Map.has_key?(tool.parameters, "additionalProperties")
    end

    test "given a raw map, then it is passed through and :schema stays nil" do
      params = %{"type" => "object", "properties" => %{"x" => %{"type" => "string"}}}

      assert {:ok, tool} =
               Tool.new(
                 name: "lookup",
                 description: "d",
                 parameters: params,
                 function: fn _args -> :ok end
               )

      assert tool.parameters == params
      assert tool.schema == nil
    end

    test "given a nested schema, then embeds appear in the parameters" do
      assert {:ok, tool} =
               Tool.new(
                 name: "save",
                 description: "d",
                 parameters: Person,
                 function: fn _args -> :ok end
               )

      assert tool.parameters["properties"]["address"]["type"] == "object"
      assert tool.parameters["properties"]["address"]["properties"]["city"]["type"] == "string"
    end
  end

  describe "Tool.new/1 failures" do
    test "given a module that is not an Ecto schema, then it is refused by name" do
      assert {:error, message} =
               Tool.new(
                 name: "lookup",
                 description: "d",
                 parameters: URI,
                 function: fn _args -> :ok end
               )

      assert message =~ "URI is not an Ecto schema"
    end

    test "given a schema with an inexpressible field, then the field is named" do
      assert {:error, message} =
               Tool.new(
                 name: "lookup",
                 description: "d",
                 parameters: ExAgent.Test.Schemas.FreeMap,
                 function: fn _args -> :ok end
               )

      assert message =~ ":payload"
    end

    test "given a schema with no fields, then it is refused" do
      assert {:error, message} =
               Tool.new(
                 name: "lookup",
                 description: "d",
                 parameters: ExAgent.Test.Schemas.Empty,
                 function: fn _args -> :ok end
               )

      assert message =~ "no fields"
    end
  end

  describe "execution" do
    test "given a typed tool, then the function receives a struct" do
      test_pid = self()

      {:ok, tool} =
        Tool.new(
          name: "lookup",
          description: "d",
          parameters: Query,
          function: fn args ->
            send(test_pid, {:args, args})
            {:ok, "found"}
          end
        )

      {:ok, agent} =
        ExAgent.start_agent(
          provider: provider(tool_calling_plug(~s({"order_id":"A-1042"}))),
          tools: [tool]
        )

      assert {:ok, _response} = ExAgent.chat(agent, "Where is A-1042?")
      assert_received {:args, %Query{order_id: "A-1042"}}
    end

    test "given args that do not fit, then the model is told rather than crashed" do
      test_pid = self()

      {:ok, tool} =
        Tool.new(
          name: "lookup",
          description: "d",
          parameters: Person,
          function: fn args ->
            send(test_pid, {:called, args})
            {:ok, "found"}
          end
        )

      {:ok, agent} =
        ExAgent.start_agent(
          provider: provider(tool_calling_plug(~s({"name":"Ada"}))),
          tools: [tool]
        )

      # The turn completes: the cast failure is fed back as a tool error, which is
      # what lets the model correct itself.
      assert {:ok, response} = ExAgent.chat(agent, "Save Ada")
      assert response.content == "done"
      refute_received {:called, _args}

      tool_message =
        agent |> ExAgent.get_context() |> Map.fetch!(:messages) |> Enum.find(&(&1.role == :tool))

      assert tool_message.content =~ "age"
    end

    test "given an untyped tool, then the function still receives a plain map" do
      test_pid = self()

      {:ok, tool} =
        Tool.new(
          name: "lookup",
          description: "d",
          parameters: %{"type" => "object", "properties" => %{}},
          function: fn args ->
            send(test_pid, {:args, args})
            {:ok, "found"}
          end
        )

      {:ok, agent} =
        ExAgent.start_agent(
          provider: provider(tool_calling_plug(~s({"order_id":"A-1042"}))),
          tools: [tool]
        )

      assert {:ok, _response} = ExAgent.chat(agent, "go")
      assert_received {:args, %{"order_id" => "A-1042"}}
    end
  end

  describe "edge cases" do
    test "given a typed tool with an embeds_one, then the inner struct is built" do
      test_pid = self()

      {:ok, tool} =
        Tool.new(
          name: "lookup",
          description: "d",
          parameters: Person,
          function: fn args ->
            send(test_pid, {:args, args})
            {:ok, "ok"}
          end
        )

      args =
        Jason.encode!(%{
          "name" => "Ada",
          "age" => 36,
          "active" => true,
          "tags" => ["math"],
          "seen_at" => "2026-03-14T10:00:00Z",
          "address" => %{"city" => "London"}
        })

      {:ok, agent} =
        ExAgent.start_agent(provider: provider(tool_calling_plug(args)), tools: [tool])

      assert {:ok, _response} = ExAgent.chat(agent, "go")
      assert_received {:args, %Person{address: %Address{city: "London"}}}
    end

    test "given a schema module, then the wire tool declaration uses the generated schema" do
      test_pid = self()

      plug = fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        send(test_pid, {:request, Jason.decode!(body)})
        Req.Test.json(conn, %{"choices" => [%{"message" => %{"content" => "ok"}}]})
      end

      {:ok, tool} =
        Tool.new(
          name: "lookup",
          description: "Look up an order",
          parameters: Query,
          function: fn _args -> :ok end
        )

      {:ok, agent} = ExAgent.start_agent(provider: provider(plug), tools: [tool])
      assert {:ok, _r} = ExAgent.chat(agent, "go")

      assert_received {:request, body}
      [declared] = body["tools"]
      assert declared["function"]["parameters"]["required"] == ["order_id"]
    end
  end
end
