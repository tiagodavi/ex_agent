defmodule ExAgent.Providers.GeminiTest do
  use ExUnit.Case, async: true

  alias ExAgent.Providers.Gemini

  # Happy path tests
  describe "new/1" do
    test "creates a provider with required api_key" do
      provider = Gemini.new(api_key: "AIza-test-key")
      assert provider.api_key == "AIza-test-key"
      assert provider.model == "gemini-2.0-flash"
      assert %Req.Request{} = provider.req
    end

    test "accepts custom model" do
      provider = Gemini.new(api_key: "AIza-test", model: "gemini-1.5-pro")
      assert provider.model == "gemini-1.5-pro"
    end

    test "accepts system_prompt" do
      provider = Gemini.new(api_key: "AIza-test", system_prompt: "Be concise")
      assert provider.system_prompt == "Be concise"
    end
  end

  # Bad path tests
  describe "new/1 validation errors" do
    test "raises on missing api_key" do
      assert_raise NimbleOptions.ValidationError, fn ->
        Gemini.new([])
      end
    end

    test "raises on invalid model type" do
      assert_raise NimbleOptions.ValidationError, fn ->
        Gemini.new(api_key: "AIza-test", model: 123)
      end
    end

    test "raises on invalid base_url type" do
      assert_raise NimbleOptions.ValidationError, fn ->
        Gemini.new(api_key: "AIza-test", base_url: 123)
      end
    end
  end

  # Edge case tests
  describe "new/1 edge cases" do
    test "initializes req client with x-goog-api-key header" do
      provider = Gemini.new(api_key: "AIza-test-key")
      headers = provider.req.headers
      assert Map.has_key?(headers, "x-goog-api-key")
      assert headers["x-goog-api-key"] == ["AIza-test-key"]
    end

    test "uses correct base_url for Gemini API" do
      provider = Gemini.new(api_key: "AIza-test")
      assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"
    end

    test "defaults tools to empty list" do
      provider = Gemini.new(api_key: "AIza-test")
      assert provider.tools == []
    end
  end
end
