defmodule ExAgent.CollectTest do
  @moduledoc """
  `ExAgent.collect/1` must fold a chunk stream into the same `ExAgent.Response`
  that `ExAgent.chat/3` returns, so streaming and non-streaming share one
  downstream code path.
  """

  use ExUnit.Case, async: true

  alias ExAgent.{Chunk, Error, Message, Response}
  alias ExAgent.Providers.OpenAI
  alias ExAgent.Services.OpenAIService
  alias ExAgent.Test.FakeSSE

  defp sse(frames) do
    Enum.map_join(frames, "", fn f -> "data: #{Jason.encode!(f)}\n\n" end) <> "data: [DONE]\n\n"
  end

  defp provider(parts, opts \\ []) do
    %OpenAI{
      api_key: "sk-test",
      model: "gpt-4o",
      base_url: "http://x",
      req: Req.new(adapter: FakeSSE.adapter(parts, opts))
    }
  end

  defp user_msg do
    {:ok, msg} = Message.new(role: :user, content: "hi")
    msg
  end

  defp texts(chunks),
    do: chunks |> Enum.filter(&(&1.type == :text_delta)) |> Enum.map(& &1.text)

  describe "collect/1 on chunk lists" do
    test "given text deltas, then they concatenate in order" do
      chunks = [Chunk.text_delta("Hello"), Chunk.text_delta(" "), Chunk.text_delta("world")]

      assert {:ok, %Response{content: "Hello world"}} = ExAgent.collect(chunks)
    end

    test "given a done chunk, then the finish reason is carried" do
      chunks = [Chunk.text_delta("hi"), Chunk.done(:length)]

      assert {:ok, %Response{finish_reason: :length}} = ExAgent.collect(chunks)
    end

    test "given usage chunks, then they are merged" do
      chunks = [
        Chunk.text_delta("hi"),
        Chunk.usage(%{input_tokens: 10, output_tokens: 5, total_tokens: 15}),
        Chunk.done()
      ]

      assert {:ok, %Response{usage: usage}} = ExAgent.collect(chunks)
      assert usage.total_tokens == 15
    end

    test "given thinking deltas, then they land in :thinking, not :content" do
      chunks = [
        Chunk.thinking_delta("Let me reason. "),
        Chunk.thinking_delta("Done."),
        Chunk.text_delta("The answer is 4."),
        Chunk.done()
      ]

      assert {:ok, response} = ExAgent.collect(chunks)
      assert response.content == "The answer is 4."
      assert response.thinking == "Let me reason. Done."
      # Reasoning must not leak into conversation history.
      refute response.message.content =~ "reason"
    end

    test "given no thinking, then :thinking is nil" do
      assert {:ok, %Response{thinking: nil}} = ExAgent.collect([Chunk.text_delta("x")])
    end

    test "given fragmented tool calls, then they are reassembled" do
      chunks = [
        %Chunk{type: :tool_call_delta, index: 0, name: "search", arguments: ~s({"q":)},
        %Chunk{type: :tool_call_delta, index: 0, arguments: ~s("elixir"})},
        Chunk.done(:tool_calls)
      ]

      assert {:ok, response} = ExAgent.collect(chunks)
      assert response.tool_calls == [%{"name" => "search", "args" => %{"q" => "elixir"}}]
      assert response.finish_reason == :tool_calls
    end

    test "given an error chunk, then collect returns the error" do
      error = Error.new(:rate_limit, "slow down")

      assert {:error, ^error} = ExAgent.collect([Chunk.text_delta("partial"), Chunk.error(error)])
    end

    test "given an empty stream, then an empty response comes back" do
      assert {:ok, %Response{content: "", tool_calls: nil}} = ExAgent.collect([])
    end

    test "given a response, then :message mirrors :content for context storage" do
      assert {:ok, response} = ExAgent.collect([Chunk.text_delta("stored")])

      assert %Message{role: :assistant, content: "stored"} = response.message
    end
  end

  describe "collect/1 over a real stream" do
    test "given a streamed turn, then it equals what chat/3 would have returned" do
      frames = [
        %{"choices" => [%{"delta" => %{"content" => "Hello"}}]},
        %{"choices" => [%{"delta" => %{"content" => " world"}}, %{}]},
        %{"choices" => [%{"delta" => %{}, "finish_reason" => "stop"}]},
        %{
          "choices" => [],
          "usage" => %{"prompt_tokens" => 3, "completion_tokens" => 2, "total_tokens" => 5}
        }
      ]

      stream = OpenAIService.stream(provider([sse(frames)]), [user_msg()])

      assert {:ok, response} = ExAgent.collect(stream)
      assert response.content == "Hello world"
      assert response.finish_reason == :stop
      assert response.usage.total_tokens == 5
    end

    test "given a mid-stream transport failure, then partial text is discarded for the error" do
      body = sse([%{"choices" => [%{"delta" => %{"content" => "partial"}}]}])
      # No DONE: the connection drops instead.
      body = String.replace(body, "data: [DONE]\n\n", "")

      stream = OpenAIService.stream(provider([body], error: :closed), [user_msg()])
      chunks = Enum.to_list(stream)

      # The text that did arrive is still emitted...
      assert Enum.any?(chunks, &(&1.type == :text_delta and &1.text == "partial"))

      # ...and the failure arrives as the terminal chunk, not a raise.
      assert %Chunk{type: :done, finish_reason: :error, error: error} = List.last(chunks)
      assert error.type == :transport
      assert error.retryable?

      assert {:error, %Error{type: :transport}} = ExAgent.collect(chunks)
    end

    test "given an HTTP 429 before any data, then one error chunk is produced" do
      stream =
        OpenAIService.stream(
          provider([~s({"error":{"message":"slow down"}})], status: 429),
          [user_msg()]
        )

      assert [%Chunk{type: :done, finish_reason: :error, error: error}] = Enum.to_list(stream)
      assert error.type == :rate_limit
      assert error.retryable?
      assert error.message == "slow down"
    end
  end

  describe "parity with chat/3" do
    # Task 7's acceptance criterion: collecting a stream must produce the same
    # response a blocking call would have, so downstream code is identical.
    test "given the same turn, then collect/1 equals chat/3" do
      text = "Elixir is a functional language."

      usage = %{"prompt_tokens" => 5, "completion_tokens" => 7, "total_tokens" => 12}

      blocking =
        Req.new(
          plug: fn conn ->
            Req.Test.json(conn, %{
              "choices" => [
                %{
                  "message" => %{"role" => "assistant", "content" => text},
                  "finish_reason" => "stop"
                }
              ],
              "usage" => usage
            })
          end
        )

      streamed =
        sse([
          %{"choices" => [%{"delta" => %{"content" => "Elixir is a "}}]},
          %{"choices" => [%{"delta" => %{"content" => "functional language."}}]},
          %{"choices" => [%{"delta" => %{}, "finish_reason" => "stop"}]},
          %{"choices" => [], "usage" => usage}
        ])

      chat_provider = %OpenAI{
        api_key: "sk-test",
        model: "gpt-4o",
        base_url: "http://x",
        req: blocking
      }

      assert {:ok, from_chat} = OpenAIService.chat(chat_provider, [user_msg()])

      assert {:ok, from_stream} =
               provider([streamed]) |> OpenAIService.stream([user_msg()]) |> ExAgent.collect()

      assert from_stream.content == from_chat.content
      assert from_stream.finish_reason == from_chat.finish_reason
      assert from_stream.usage == from_chat.usage
      assert from_stream.tool_calls == from_chat.tool_calls
      assert from_stream.message.role == from_chat.message.role
      assert from_stream.message.content == from_chat.message.content
    end
  end

  describe "stream invariants" do
    test "given any successful stream, then it ends with exactly one done chunk" do
      frames = [%{"choices" => [%{"delta" => %{"content" => "a"}}]}]
      chunks = provider([sse(frames)]) |> OpenAIService.stream([user_msg()]) |> Enum.to_list()

      assert Enum.count(chunks, &(&1.type == :done)) == 1
      assert List.last(chunks).type == :done
    end

    # A provider mapper turns a finish-reason frame into its own :done chunk, and
    # the transport appends the terminal one. Without dedup that is two, which
    # breaks the invariant for every real response - models always finish.
    test "given a frame carrying a finish reason, then there is still exactly one done chunk" do
      frames = [
        %{"choices" => [%{"delta" => %{"content" => "a"}}]},
        %{"choices" => [%{"delta" => %{}, "finish_reason" => "stop"}]}
      ]

      chunks = provider([sse(frames)]) |> OpenAIService.stream([user_msg()]) |> Enum.to_list()

      assert Enum.count(chunks, &(&1.type == :done)) == 1
      assert List.last(chunks).finish_reason == :stop
    end

    test "given a finish reason arriving in its own read, then it still yields one done chunk" do
      # Separate reads exercise the path where the mapper's :done is the only
      # chunk produced for that frame.
      parts = [
        ~s(data: {"choices":[{"delta":{"content":"a"}}]}\n\n),
        ~s(data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n),
        "data: [DONE]\n\n"
      ]

      chunks = provider(parts) |> OpenAIService.stream([user_msg()]) |> Enum.to_list()

      assert Enum.count(chunks, &(&1.type == :done)) == 1
      assert List.last(chunks).finish_reason == :length
      assert texts(chunks) == ["a"]
    end

    test "given a failing stream, then it also ends with exactly one done chunk" do
      chunks =
        provider(["{}"], status: 500)
        |> OpenAIService.stream([user_msg()])
        |> Enum.to_list()

      assert Enum.count(chunks, &(&1.type == :done)) == 1
      assert List.last(chunks).type == :done
    end
  end
end
