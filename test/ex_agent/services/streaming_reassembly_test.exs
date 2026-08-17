defmodule ExAgent.Services.StreamingReassemblyTest do
  @moduledoc """
  Exercises `ExAgent.Services.Streaming` against a body that arrives in multiple
  pieces, which the plug-based stubs cannot produce.
  """

  use ExUnit.Case, async: true

  alias ExAgent.Message
  alias ExAgent.Providers.OpenAI
  alias ExAgent.Test.FakeSSE

  defp frame(content), do: %{"choices" => [%{"delta" => %{"content" => content}}]}

  defp sse(frames) do
    Enum.map_join(frames, "", fn f -> "data: #{Jason.encode!(f)}\n\n" end) <> "data: [DONE]\n\n"
  end

  defp provider(parts, opts) do
    %OpenAI{
      api_key: "sk-test",
      model: "gpt-4o",
      base_url: "http://x",
      req: FakeSSE.req(parts, opts)
    }
  end

  defp user_msg do
    {:ok, msg} = Message.new(role: :user, content: "hi")
    msg
  end

  defp run(parts, opts \\ []) do
    parts
    |> provider(opts)
    |> OpenAI.stream([user_msg()])
    |> Enum.filter(&(&1.type == :text_delta))
    |> Enum.map(& &1.text)
  end

  describe "frames split across reads" do
    test "given a body delivered whole, then all chunks arrive" do
      body = sse([frame("Hello"), frame(" world")])

      assert ["Hello", " world"] = run([body])
    end

    test "given a body split at every byte boundary, then the text is always identical" do
      body = sse([frame("Hello"), frame(" world")])

      for parts <- FakeSSE.all_splits(body) do
        assert Enum.join(run(parts)) == "Hello world",
               "reassembly failed for split #{inspect(Enum.map(parts, &byte_size/1))}"
      end
    end

    test "given a body arriving in many small slices, then every frame is recovered" do
      body = sse(Enum.map(1..8, &frame("chunk#{&1}")))
      parts = FakeSSE.split(body, 20)

      assert Enum.join(run(parts)) == Enum.map_join(1..8, "", &"chunk#{&1}")
    end

    test "given a single frame split mid-JSON, then it is not emitted twice" do
      body = sse([frame("once")])
      midpoint = div(byte_size(body), 2)
      <<first::binary-size(midpoint), second::binary>> = body

      assert ["once"] = run([first, second])
    end
  end

  describe "termination" do
    test "given a DONE sentinel split from its frame, then the stream still ends" do
      body = sse([frame("text")])
      at = byte_size(body) - 8
      <<first::binary-size(at), second::binary>> = body

      assert ["text"] = run([first, second])
    end

    test "given no DONE sentinel, then the transport close ends the stream" do
      body = "data: #{Jason.encode!(frame("no sentinel"))}\n\n"

      assert ["no sentinel"] = run([body])
    end
  end
end
