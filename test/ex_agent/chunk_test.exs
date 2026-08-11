defmodule ExAgent.ChunkTest do
  use ExUnit.Case, async: true

  doctest ExAgent.Chunk

  alias ExAgent.{Chunk, Error}

  describe "constructors" do
    test "given text, then a text delta is built" do
      assert %Chunk{type: :text_delta, text: "hi"} = Chunk.text_delta("hi")
    end

    test "given reasoning text, then a thinking delta is built" do
      assert %Chunk{type: :thinking_delta, text: "hmm"} = Chunk.thinking_delta("hmm")
    end

    test "given usage, then a usage chunk is built" do
      assert %Chunk{type: :usage, usage: %{total_tokens: 10}} = Chunk.usage(%{total_tokens: 10})
    end

    test "given no arguments, then done defaults to :stop" do
      assert %Chunk{type: :done, finish_reason: :stop, error: nil} = Chunk.done()
    end

    test "given an error, then the terminal chunk carries it" do
      error = Error.new(:rate_limit, "slow down")

      assert %Chunk{type: :done, finish_reason: :error, error: ^error} = Chunk.error(error)
    end
  end

  describe "finish_reason/1" do
    test "given known reasons, then they normalize" do
      assert Chunk.finish_reason("stop") == :stop
      assert Chunk.finish_reason("length") == :length
      assert Chunk.finish_reason("tool_calls") == :tool_calls
      assert Chunk.finish_reason("content_filter") == :content_filter
    end

    test "given provider-specific spellings, then they still normalize" do
      assert Chunk.finish_reason("MAX_TOKENS") == :length
      assert Chunk.finish_reason("STOP") == :stop
      assert Chunk.finish_reason("SAFETY") == :content_filter
      assert Chunk.finish_reason("end_turn") == :stop
    end

    test "given nil, then it stays nil" do
      assert Chunk.finish_reason(nil) == nil
    end

    test "given an unknown reason, then it falls back to :stop rather than vanishing" do
      assert Chunk.finish_reason("something_new") == :stop
    end
  end

  describe "finalize_tool_calls/1" do
    test "given no chunks, then nil is returned so Message.new accepts it" do
      assert Chunk.finalize_tool_calls([]) == nil
    end

    test "given only non-tool chunks, then nil is returned" do
      assert Chunk.finalize_tool_calls([Chunk.text_delta("hi"), Chunk.done()]) == nil
    end

    test "given fragments of one call, then arguments are concatenated and decoded" do
      chunks = [
        %Chunk{type: :tool_call_delta, index: 0, name: "search", arguments: ~s({"q":)},
        %Chunk{type: :tool_call_delta, index: 0, arguments: ~s("elixir")},
        %Chunk{type: :tool_call_delta, index: 0, arguments: "}"}
      ]

      assert [%{"name" => "search", "args" => %{"q" => "elixir"}}] =
               Chunk.finalize_tool_calls(chunks)
    end

    test "given interleaved calls, then each is assembled by index and ordered" do
      chunks = [
        %Chunk{type: :tool_call_delta, index: 0, name: "a", arguments: ~s({"x":)},
        %Chunk{type: :tool_call_delta, index: 1, name: "b", arguments: ~s({"y":)},
        %Chunk{type: :tool_call_delta, index: 0, arguments: "1}"},
        %Chunk{type: :tool_call_delta, index: 1, arguments: "2}"}
      ]

      assert [
               %{"name" => "a", "args" => %{"x" => 1}},
               %{"name" => "b", "args" => %{"y" => 2}}
             ] = Chunk.finalize_tool_calls(chunks)
    end

    test "given a name arriving only on the first fragment, then it is still found" do
      chunks = [
        %Chunk{type: :tool_call_delta, index: 0, name: "once", arguments: "{}"},
        %Chunk{type: :tool_call_delta, index: 0, name: nil, arguments: ""}
      ]

      assert [%{"name" => "once"}] = Chunk.finalize_tool_calls(chunks)
    end

    test "given arguments that never form valid JSON, then the raw text is kept" do
      chunks = [%Chunk{type: :tool_call_delta, index: 0, name: "x", arguments: "not json"}]

      assert [%{"name" => "x", "args" => %{"raw" => "not json"}}] =
               Chunk.finalize_tool_calls(chunks)
    end

    test "given empty arguments, then args is an empty map" do
      chunks = [%Chunk{type: :tool_call_delta, index: 0, name: "x", arguments: nil}]

      assert [%{"name" => "x", "args" => %{}}] = Chunk.finalize_tool_calls(chunks)
    end
  end
end
