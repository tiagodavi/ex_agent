defmodule ExAgent.Providers.DeepSeekTest do
  use ExUnit.Case, async: true

  alias ExAgent.Providers.DeepSeek

  # Happy path tests
  describe "new/1" do
    test "creates a provider with required api_key" do
      provider = DeepSeek.new(api_key: "sk-deep-test")
      assert provider.api_key == "sk-deep-test"
      assert provider.model == "deepseek-chat"
      assert provider.base_url == "https://api.deepseek.com/v1"
      assert %Req.Request{} = provider.req
    end

    test "accepts custom model" do
      provider = DeepSeek.new(api_key: "sk-deep", model: "deepseek-reasoner")
      assert provider.model == "deepseek-reasoner"
    end

    test "accepts system_prompt" do
      provider = DeepSeek.new(api_key: "sk-deep", system_prompt: "You are a coder")
      assert provider.system_prompt == "You are a coder"
    end
  end

  # Bad path tests
  describe "new/1 validation errors" do
    test "raises on missing api_key" do
      assert_raise NimbleOptions.ValidationError, fn ->
        DeepSeek.new([])
      end
    end

    test "raises on invalid model type" do
      assert_raise NimbleOptions.ValidationError, fn ->
        DeepSeek.new(api_key: "sk-deep", model: 123)
      end
    end

    test "raises on invalid base_url type" do
      assert_raise NimbleOptions.ValidationError, fn ->
        DeepSeek.new(api_key: "sk-deep", base_url: 123)
      end
    end
  end

  # Edge case tests
  describe "new/1 edge cases" do
    test "initializes req client with Bearer auth header" do
      provider = DeepSeek.new(api_key: "sk-deep-key")
      headers = provider.req.headers
      assert Map.has_key?(headers, "authorization")
      assert headers["authorization"] == ["Bearer sk-deep-key"]
    end

    test "defaults tools to empty list" do
      provider = DeepSeek.new(api_key: "sk-deep")
      assert provider.tools == []
    end

    test "defaults system_prompt to nil" do
      provider = DeepSeek.new(api_key: "sk-deep")
      assert provider.system_prompt == nil
    end
  end
end
