defmodule ExAgent.ToolTest do
  use ExUnit.Case, async: true

  alias ExAgent.Tool

  doctest ExAgent.Tool

  # Happy path tests
  describe "new/1 with valid attrs" do
    test "creates a tool with all required fields" do
      assert {:ok, %Tool{name: "search", description: "Search the web"}} =
               Tool.new(
                 name: "search",
                 description: "Search the web",
                 parameters: %{"type" => "object", "properties" => %{}},
                 function: fn _args -> {:ok, "result"} end
               )
    end

    test "stores the function for later execution" do
      fun = fn args -> {:ok, Map.get(args, "query")} end

      {:ok, tool} =
        Tool.new(
          name: "search",
          description: "Search",
          parameters: %{},
          function: fun
        )

      assert {:ok, "elixir"} = tool.function.(%{"query" => "elixir"})
    end

    test "stores JSON Schema parameters" do
      params = %{
        "type" => "object",
        "properties" => %{
          "query" => %{"type" => "string", "description" => "Search query"}
        },
        "required" => ["query"]
      }

      {:ok, tool} =
        Tool.new(name: "search", description: "Search", parameters: params, function: & &1)

      assert tool.parameters == params
    end
  end

  # Bad path tests
  describe "new/1 with invalid attrs" do
    test "returns error for missing name" do
      assert {:error, _reason} =
               Tool.new(description: "Search", parameters: %{}, function: & &1)
    end

    test "returns error for missing description" do
      assert {:error, _reason} =
               Tool.new(name: "search", parameters: %{}, function: & &1)
    end

    test "returns error for missing function" do
      assert {:error, _reason} =
               Tool.new(name: "search", description: "Search", parameters: %{})
    end
  end

  # Edge case tests
  describe "new/1 edge cases" do
    test "accepts empty parameters map" do
      assert {:ok, %Tool{parameters: %{}}} =
               Tool.new(name: "noop", description: "No-op", parameters: %{}, function: fn _ -> :ok end)
    end

    test "function can return error tuples" do
      fun = fn _args -> {:error, "not found"} end

      {:ok, tool} =
        Tool.new(name: "search", description: "Search", parameters: %{}, function: fun)

      assert {:error, "not found"} = tool.function.(%{})
    end

    test "accepts missing parameters defaulting to empty map" do
      assert {:ok, %Tool{parameters: %{}}} =
               Tool.new(name: "noop", description: "No-op", function: fn _ -> :ok end)
    end
  end
end
