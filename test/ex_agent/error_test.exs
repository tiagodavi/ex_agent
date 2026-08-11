defmodule ExAgent.ErrorTest do
  use ExUnit.Case, async: true

  alias ExAgent.Error
  alias ExAgent.Providers.OpenAI

  defp openai_body(message, extra \\ %{}) do
    %{"error" => Map.merge(%{"message" => message}, extra)}
  end

  describe "from_http/3 classification" do
    test "given a 401, when classified, then it is a non-retryable auth error" do
      error = Error.from_http(401, openai_body("Invalid API key"), OpenAI)

      assert error.type == :auth
      assert error.status == 401
      assert error.provider == OpenAI
      refute error.retryable?
    end

    test "given a 403, when classified, then it is a non-retryable auth error" do
      error = Error.from_http(403, openai_body("Forbidden"), OpenAI)

      assert error.type == :auth
      refute error.retryable?
    end

    test "given a 404, when classified, then it is a non-retryable not_found error" do
      error = Error.from_http(404, openai_body("No such model"), OpenAI)

      assert error.type == :not_found
      refute error.retryable?
    end

    test "given a 408, when classified, then it is a retryable timeout error" do
      error = Error.from_http(408, openai_body("Request timeout"), OpenAI)

      assert error.type == :timeout
      assert error.retryable?
    end

    test "given a 429, when classified, then it is a retryable rate_limit error" do
      error = Error.from_http(429, openai_body("Rate limit reached"), OpenAI)

      assert error.type == :rate_limit
      assert error.retryable?
    end

    test "given a plain 400, when classified, then it is a non-retryable invalid_request" do
      error = Error.from_http(400, openai_body("Unknown parameter: foo"), OpenAI)

      assert error.type == :invalid_request
      refute error.retryable?
    end

    test "given a 400 with a context_length_exceeded code, then it is a context_length error" do
      body =
        openai_body("This model's maximum context length is 8192 tokens", %{
          "code" => "context_length_exceeded"
        })

      error = Error.from_http(400, body, OpenAI)

      assert error.type == :context_length
      refute error.retryable?
    end

    test "given a 400 whose message mentions context length, then it is a context_length error" do
      error =
        Error.from_http(
          400,
          openai_body("The input token count exceeds the maximum context length"),
          OpenAI
        )

      assert error.type == :context_length
    end

    test "given a 500, when classified, then it is a retryable server error" do
      error = Error.from_http(500, openai_body("Internal error"), OpenAI)

      assert error.type == :server
      assert error.retryable?
    end

    test "given a 503, when classified, then it is a retryable server error" do
      error = Error.from_http(503, "Service Unavailable", OpenAI)

      assert error.type == :server
      assert error.retryable?
    end

    test "given an unmapped 4xx, when classified, then it falls back to invalid_request" do
      error = Error.from_http(422, openai_body("Unprocessable"), OpenAI)

      assert error.type == :invalid_request
      refute error.retryable?
    end
  end

  describe "from_http/3 message extraction" do
    test "given a nested error.message body, then that message is used" do
      error = Error.from_http(400, openai_body("Invalid value for temperature"), OpenAI)

      assert error.message == "Invalid value for temperature"
    end

    test "given a flat message body, then that message is used" do
      error = Error.from_http(400, %{"message" => "Bad request"}, OpenAI)

      assert error.message == "Bad request"
    end

    test "given a plain string body, then the string is used" do
      error = Error.from_http(500, "upstream exploded", OpenAI)

      assert error.message == "upstream exploded"
    end

    test "given an unrecognised body shape, then it falls back to the status line" do
      error = Error.from_http(500, %{"weird" => [1, 2, 3]}, OpenAI)

      assert error.message == "HTTP 500"
    end

    test "given any body, then raw preserves it verbatim for incremental migration" do
      body = openai_body("Rate limit reached")
      error = Error.from_http(429, body, OpenAI)

      assert error.raw == body
    end
  end

  describe "from_transport/2" do
    test "given a transport timeout, then it is a retryable timeout error" do
      error = Error.from_transport(%Req.TransportError{reason: :timeout}, OpenAI)

      assert error.type == :timeout
      assert error.retryable?
      assert error.status == nil
      assert error.provider == OpenAI
    end

    test "given a refused connection, then it is a retryable transport error" do
      error = Error.from_transport(%Req.TransportError{reason: :econnrefused}, OpenAI)

      assert error.type == :transport
      assert error.retryable?
    end

    test "given an arbitrary reason, then it is a retryable transport error" do
      error = Error.from_transport(:closed, OpenAI)

      assert error.type == :transport
      assert error.retryable?
      assert error.raw == :closed
    end
  end

  describe "new/3" do
    test "given a type and message, then it builds a non-retryable error" do
      error = Error.new(:unsupported, "provider does not accept image input", OpenAI)

      assert error.type == :unsupported
      assert error.message == "provider does not accept image input"
      assert error.provider == OpenAI
      assert error.status == nil
      refute error.retryable?
    end

    test "given no provider, then provider is nil" do
      assert Error.new(:invalid_request, "boom").provider == nil
    end
  end

  describe "from_result/2" do
    test "given a 200 response, then the body passes through untouched" do
      body = %{"choices" => []}

      assert {:ok, ^body} =
               Error.from_result({:ok, %Req.Response{status: 200, body: body}}, OpenAI)
    end

    test "given a non-200 response, then it maps to an %Error{}" do
      assert {:error, %Error{type: :rate_limit, status: 429}} =
               Error.from_result(
                 {:ok, %Req.Response{status: 429, body: openai_body("slow down")}},
                 OpenAI
               )
    end

    test "given a transport failure, then it maps to an %Error{}" do
      assert {:error, %Error{type: :timeout, retryable?: true}} =
               Error.from_result({:error, %Req.TransportError{reason: :timeout}}, OpenAI)
    end
  end

  # Integration: every service must surface %Error{} while preserving the
  # original body in :raw, so callers pattern-matching on it can migrate
  # incrementally.
  describe "service integration" do
    setup do
      body = %{"error" => %{"message" => "Rate limit reached", "type" => "requests"}}

      plug = fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.send_resp(429, Jason.encode!(body))
      end

      {:ok, message} = ExAgent.Message.new(role: :user, content: "Hi")

      %{body: body, req: Req.new(plug: plug), message: message}
    end

    test "given a 429, when OpenAI chats, then it returns an %Error{} keeping raw",
         %{body: body, req: req, message: message} do
      provider = %ExAgent.Providers.OpenAI{api_key: "sk-test", model: "gpt-4o", req: req}

      assert {:error, %Error{type: :rate_limit, status: 429, retryable?: true} = error} =
               ExAgent.Services.OpenAIService.chat(provider, [message])

      assert error.provider == ExAgent.Providers.OpenAI
      assert error.raw == body
      assert error.message == "Rate limit reached"
    end

    test "given a 429, when Gemini chats, then it returns an %Error{} keeping raw",
         %{body: body, req: req, message: message} do
      provider = %ExAgent.Providers.Gemini{api_key: "AIza", model: "gemini-2.0-flash", req: req}

      assert {:error, %Error{type: :rate_limit, status: 429} = error} =
               ExAgent.Services.GeminiService.chat(provider, [message])

      assert error.provider == ExAgent.Providers.Gemini
      assert error.raw == body
    end

    test "given a provider without upload/4, when uploading, then it returns :unsupported" do
      provider = ExAgent.Test.MinimalProvider.new()

      assert {:error, %Error{type: :unsupported} = error} =
               ExAgent.Provider.upload(provider, "data", "text/plain")

      assert error.provider == ExAgent.Test.MinimalProvider
      assert error.message =~ "does not support file uploads"
      refute error.retryable?
    end

    test "given an unreadable path, when uploading a file, then it returns :invalid_request" do
      provider = ExAgent.Providers.OpenAI.new(api_key: "sk-test")

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.upload_file(provider, "/nonexistent/nope.pdf", "application/pdf")

      assert error.message =~ "failed to read file /nonexistent/nope.pdf"
      assert error.provider == ExAgent.Providers.OpenAI
    end
  end

  describe "exception behaviour" do
    test "given an error, then it can be raised and carries its message" do
      error = Error.new(:unsupported, "no streaming here", OpenAI)

      assert_raise ExAgent.Error, "no streaming here", fn -> raise error end
    end
  end
end
