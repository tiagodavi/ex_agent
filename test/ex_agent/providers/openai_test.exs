defmodule ExAgent.Providers.OpenAITest do
  use ExUnit.Case, async: true

  alias ExAgent.Providers.OpenAI

  # Happy path tests
  describe "new/1" do
    test "creates a provider with required api_key" do
      provider = OpenAI.new(api_key: "sk-test-key")
      assert provider.api_key == "sk-test-key"
      assert provider.model == "gpt-4o"
      assert provider.base_url == "https://api.openai.com/v1"
      assert %Req.Request{} = provider.req
    end

    test "accepts custom model and base_url" do
      provider = OpenAI.new(api_key: "sk-test", model: "gpt-4o-mini", base_url: "https://custom.api.com")
      assert provider.model == "gpt-4o-mini"
      assert provider.base_url == "https://custom.api.com"
    end

    test "accepts system_prompt and tools" do
      provider = OpenAI.new(api_key: "sk-test", system_prompt: "You are helpful")
      assert provider.system_prompt == "You are helpful"
    end
  end

  # Bad path tests
  describe "new/1 validation errors" do
    test "raises on missing api_key" do
      assert_raise NimbleOptions.ValidationError, fn ->
        OpenAI.new([])
      end
    end

    test "raises on invalid model type" do
      assert_raise NimbleOptions.ValidationError, fn ->
        OpenAI.new(api_key: "sk-test", model: 123)
      end
    end

    test "raises on invalid base_url type" do
      assert_raise NimbleOptions.ValidationError, fn ->
        OpenAI.new(api_key: "sk-test", base_url: 123)
      end
    end
  end

  # Edge case tests
  describe "new/1 edge cases" do
    test "initializes req client with correct auth header" do
      provider = OpenAI.new(api_key: "sk-test-key")
      headers = provider.req.headers
      assert Map.has_key?(headers, "authorization")
      assert headers["authorization"] == ["Bearer sk-test-key"]
    end

    test "defaults tools to empty list" do
      provider = OpenAI.new(api_key: "sk-test")
      assert provider.tools == []
    end

    test "defaults system_prompt to nil" do
      provider = OpenAI.new(api_key: "sk-test")
      assert provider.system_prompt == nil
    end
  end
end
