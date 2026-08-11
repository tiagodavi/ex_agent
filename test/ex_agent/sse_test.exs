defmodule ExAgent.SSETest do
  use ExUnit.Case, async: true

  doctest ExAgent.SSE

  alias ExAgent.SSE

  describe "take_events/1" do
    test "given complete events, then it returns them with an empty remainder" do
      assert {["a", "b"], ""} = SSE.take_events("a\n\nb\n\n")
    end

    test "given a trailing partial event, then it is held back" do
      assert {["a"], "partial"} = SSE.take_events("a\n\npartial")
    end

    test "given no separator at all, then everything is the remainder" do
      assert {[], "incomplete"} = SSE.take_events("incomplete")
    end

    test "given an empty buffer, then nothing is complete" do
      assert {[], ""} = SSE.take_events("")
    end

    # Gemini terminates SSE events with CRLF. Splitting on "\n\n" alone finds no
    # boundary at all, so every frame stays in the remainder and the stream
    # silently yields nothing.
    test "given CRLF-separated events, then they are still split" do
      assert {["a", "b"], ""} = SSE.take_events("a\r\n\r\nb\r\n\r\n")
    end

    test "given bare-CR-separated events, then they are still split" do
      assert {["a", "b"], ""} = SSE.take_events("a\r\rb\r\r")
    end

    test "given mixed line endings, then every event is found" do
      assert {["a", "b", "c"], ""} = SSE.take_events("a\r\n\r\nb\n\nc\r\n\r\n")
    end

    test "given a CRLF boundary split across two reads, then it is held back intact" do
      # "\r\n\r" is not a terminator until its "\n" arrives, so it must not be
      # mistaken for a bare-CR boundary.
      assert {["a"], "\r\n\r"} = SSE.take_events("a\r\n\r\n\r\n\r")
      assert {["", "b"], ""} = SSE.take_events("\r\n\r" <> "\n" <> "b\r\n\r\n")
    end
  end

  describe "event_data/1" do
    test "given a data line, then the payload is returned" do
      assert SSE.event_data("data: hello") == "hello"
    end

    test "given multiple data lines, then they are joined" do
      assert SSE.event_data("data: one\ndata: two") == "onetwo"
    end

    test "given non-data lines, then they are ignored" do
      assert SSE.event_data("event: message\nid: 1\ndata: payload\nretry: 100") == "payload"
    end

    test "given the DONE sentinel, then :done is returned" do
      assert SSE.event_data("data: [DONE]") == :done
    end

    test "given a comment or keep-alive, then nil is returned" do
      assert SSE.event_data(": keep-alive") == nil
      assert SSE.event_data("event: ping") == nil
    end

    test "given CRLF line endings, then no carriage return leaks into the payload" do
      assert SSE.event_data("event: message\r\ndata: payload") == "payload"
      assert SSE.event_data("data: one\r\ndata: two") == "onetwo"
      assert SSE.event_data("data: [DONE]\r\n") == :done
    end
  end

  describe "decode/1" do
    test "given complete frames, then they are decoded in order" do
      buffer = ~s(data: {"n":1}\n\ndata: {"n":2}\n\n)

      assert {[%{"n" => 1}, %{"n" => 2}], ""} = SSE.decode(buffer)
    end

    test "given a DONE sentinel, then it decodes to the :done atom" do
      assert {[%{"n" => 1}, :done], ""} = SSE.decode(~s(data: {"n":1}\n\ndata: [DONE]\n\n))
    end

    test "given a partial trailing frame, then it is returned as the remainder" do
      assert {[%{"n" => 1}], ~s(data: {"n":)} = SSE.decode(~s(data: {"n":1}\n\ndata: {"n":))
    end

    test "given malformed JSON, then the frame is skipped rather than raising" do
      assert {[%{"n" => 1}], ""} = SSE.decode(~s(data: {"n":1}\n\ndata: not json\n\n))
    end

    test "given keep-alive events, then they produce no frames" do
      assert {[], ""} = SSE.decode(": keep-alive\n\n: keep-alive\n\n")
    end

    # The exact framing Gemini sends. Decoding this to [] is why live streaming
    # produced a terminal chunk and no text at all.
    test "given a CRLF-framed Gemini response, then the text frame is decoded" do
      buffer =
        ~s(data: {"candidates":[{"content":{"parts":[{"text":"12345"}],"role":"model"}}]}\r\n\r\n)

      assert {[frame], ""} = SSE.decode(buffer)

      assert get_in(frame, ["candidates", Access.at(0), "content", "parts"]) == [
               %{"text" => "12345"}
             ]
    end
  end

  describe "reassembly across reads" do
    test "given a CRLF frame split at every byte boundary, then it decodes exactly once" do
      buffer = ~s(data: {"text":"hello world"}\r\n\r\n)

      for split_at <- 1..(byte_size(buffer) - 1) do
        <<first::binary-size(split_at), second::binary>> = buffer

        {frames_a, rest} = SSE.decode(first)
        {frames_b, rest} = SSE.decode(rest <> second)

        assert frames_a ++ frames_b == [%{"text" => "hello world"}],
               "failed when split at byte #{split_at}"

        assert rest == ""
      end
    end

    test "given a frame split at every byte boundary, then it decodes exactly once" do
      buffer = ~s(data: {"text":"hello world"}\n\n)

      for split_at <- 1..(byte_size(buffer) - 1) do
        <<first::binary-size(split_at), second::binary>> = buffer

        {frames_a, rest} = SSE.decode(first)
        {frames_b, rest} = SSE.decode(rest <> second)

        assert frames_a ++ frames_b == [%{"text" => "hello world"}],
               "failed when split at byte #{split_at}"

        assert rest == ""
      end
    end

    test "given many frames arriving in arbitrary slices, then all are recovered" do
      buffer =
        1..5
        |> Enum.map_join("", fn n -> ~s(data: {"n":#{n}}\n\n) end)

      {frames, rest} =
        buffer
        |> chunk_every_bytes(7)
        |> Enum.reduce({[], ""}, fn slice, {acc, buffered} ->
          {frames, rest} = SSE.decode(buffered <> slice)
          {acc ++ frames, rest}
        end)

      assert rest == ""
      assert frames == Enum.map(1..5, &%{"n" => &1})
    end
  end

  defp chunk_every_bytes(binary, size) do
    binary
    |> :binary.bin_to_list()
    |> Enum.chunk_every(size)
    |> Enum.map(&:binary.list_to_bin/1)
  end
end
