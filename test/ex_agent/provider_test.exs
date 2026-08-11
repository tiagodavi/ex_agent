defmodule ExAgent.ProviderTest do
  use ExUnit.Case, async: true

  alias ExAgent.{Error, Message, Provider}
  alias ExAgent.Providers.{Gemini, OpenAI}
  alias ExAgent.Test.MinimalProvider

  @png <<0x89, "PNG", 0x0D, 0x0A, 0x1A, 0x0A, 0::size(64)>>

  defp openai_provider(plug) do
    %OpenAI{api_key: "sk-test", model: "gpt-4o", req: Req.new(plug: plug)}
  end

  defp gemini_provider(plug) do
    %Gemini{api_key: "AIza", model: "gemini-2.0-flash", req: Req.new(plug: plug)}
  end

  defp openai_ok_body do
    %{"choices" => [%{"message" => %{"role" => "assistant", "content" => "seen"}}]}
  end

  defp openai_ok do
    fn conn -> Req.Test.json(conn, openai_ok_body()) end
  end

  defp gemini_ok do
    fn conn ->
      Req.Test.json(conn, %{
        "candidates" => [%{"content" => %{"parts" => [%{"text" => "seen"}]}}]
      })
    end
  end

  defp message_with(files) do
    {:ok, msg} = Message.new(role: :user, content: "Look at this", attachments: files)
    msg
  end

  describe "supported_modalities/1" do
    test "given a provider without the callback, then it defaults to text only" do
      assert Provider.supported_modalities(MinimalProvider.new(api_key: "x")) == [:text]
    end

    test "given OpenAI, then it declares images and documents" do
      modalities = Provider.supported_modalities(OpenAI.new(api_key: "sk-test"))

      assert :image in modalities
      assert :document in modalities
      refute :video in modalities
      refute :audio in modalities
    end

    test "given Gemini, then it declares video and audio as well" do
      modalities = Provider.supported_modalities(Gemini.new(api_key: "AIza"))

      assert :image in modalities
      assert :document in modalities
      assert :video in modalities
      assert :audio in modalities
    end

    # Modalities are a property of the *model*, not the vendor: o1-mini takes no
    # images, and a text-only Gemini could ship tomorrow. Narrowing belongs with
    # whoever picked the model, not in a table inside the library.
    test "given a narrowed OpenAI provider, then only the declared modalities remain" do
      provider = OpenAI.new(api_key: "sk-test", model: "o1-mini", modalities: [:text])

      assert Provider.supported_modalities(provider) == [:text]
    end

    test "given a narrowed Gemini provider, then only the declared modalities remain" do
      provider = Gemini.new(api_key: "AIza", modalities: [:text, :image])

      assert Provider.supported_modalities(provider) == [:text, :image]
    end
  end

  # The agent populates :tools on the provider struct so services can read it,
  # but a provider with no tool support has no reason to carry the field - and a
  # KeyError surfacing as an opaque :server error is a poor way to say so.
  describe "providers without a :tools field" do
    test "given a chat turn, then the agent does not require the field" do
      {:ok, pid} = ExAgent.start_agent(provider: MinimalProvider.new(api_key: "x"))
      on_exit(fn -> if Process.alive?(pid), do: ExAgent.stop_agent(pid) end)

      assert {:ok, %ExAgent.Response{content: "ok"}} = ExAgent.chat(pid, "hi")
    end

    test "given a provider that does have :tools, then the agent still populates it" do
      test_pid = self()

      plug = fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        send(test_pid, {:tools, Jason.decode!(body)["tools"]})
        Req.Test.json(conn, openai_ok_body())
      end

      {:ok, tool} =
        ExAgent.Tool.new(
          name: "ping",
          description: "ping",
          parameters: %{},
          function: fn _ -> {:ok, "pong"} end
        )

      {:ok, pid} = ExAgent.start_agent(provider: openai_provider(plug), tools: [tool])
      on_exit(fn -> if Process.alive?(pid), do: ExAgent.stop_agent(pid) end)

      assert {:ok, _} = ExAgent.chat(pid, "hi")
      assert_received {:tools, [%{"function" => %{"name" => "ping"}}]}
    end
  end

  describe "chat/3 modality gating" do
    test "given an image on a text-only provider, then it returns an unsupported error rather than raising" do
      provider = MinimalProvider.new(api_key: "x")

      assert {:error, %Error{type: :unsupported} = error} =
               Provider.chat(provider, [message_with([%{data: @png}])])

      assert error.message =~ "image"
      assert error.message =~ "MinimalProvider"
      assert error.provider == MinimalProvider
      refute error.retryable?
    end

    test "given a video on OpenAI, then it returns an unsupported error" do
      provider = openai_provider(openai_ok())

      assert {:error, %Error{type: :unsupported} = error} =
               Provider.chat(provider, [message_with([%{url: "https://x.test/clip.mp4"}])])

      assert error.message =~ "video"
    end

    test "given audio on OpenAI, then it returns an unsupported error" do
      provider = openai_provider(openai_ok())

      assert {:error, %Error{type: :unsupported}} =
               Provider.chat(provider, [message_with([%{url: "https://x.test/take.mp3"}])])
    end

    test "given an image on OpenAI, then the request proceeds unchanged" do
      provider = openai_provider(openai_ok())

      assert {:ok, %ExAgent.Response{content: "seen"}} =
               Provider.chat(provider, [message_with([%{data: @png}])])
    end

    test "given a video on Gemini, then the request proceeds" do
      provider = gemini_provider(gemini_ok())

      assert {:ok, %ExAgent.Response{content: "seen"}} =
               Provider.chat(provider, [message_with([%{url: "https://x.test/clip.mp4"}])])
    end

    test "given audio on Gemini, then the request proceeds" do
      provider = gemini_provider(gemini_ok())

      assert {:ok, %ExAgent.Response{content: "seen"}} =
               Provider.chat(provider, [message_with([%{url: "https://x.test/take.wav"}])])
    end

    test "given no attachments, then a text-only provider is unaffected" do
      provider = MinimalProvider.new(api_key: "x")
      {:ok, msg} = Message.new(role: :user, content: "Hello")

      # No modality error - it fails later at the HTTP layer, not the gate.
      refute match?({:error, %Error{type: :unsupported}}, Provider.chat(provider, [msg]))
    end

    test "given an image on a narrowed OpenAI model, then it is rejected before any request" do
      # Without this the request is built happily and the API 400s instead - the
      # modality gate exists precisely so that fails here, loudly.
      provider = OpenAI.new(api_key: "sk-test", model: "o1-mini", modalities: [:text])

      assert {:error, %Error{type: :unsupported} = error} =
               Provider.chat(provider, [message_with([%{data: @png}])])

      assert error.message =~ "image"
      assert error.provider == OpenAI
    end

    test "given a narrowed Gemini provider, then video is rejected while images pass" do
      narrowed = Gemini.new(api_key: "AIza", modalities: [:text, :image])

      assert {:error, %Error{type: :unsupported}} =
               Provider.chat(narrowed, [message_with([%{url: "https://x.test/clip.mp4"}])])
    end

    test "given a supported and an unsupported attachment, then the unsupported one is reported" do
      provider = openai_provider(openai_ok())

      msg = message_with([%{data: @png}, %{url: "https://x.test/clip.mp4"}])

      assert {:error, %Error{type: :unsupported} = error} = Provider.chat(provider, [msg])
      assert error.message =~ "video"
    end
  end

  describe "stream/3 modality gating" do
    test "given an unsupported modality, then it raises an unsupported ExAgent.Error" do
      provider = MinimalProvider.new(api_key: "x")

      error =
        assert_raise Error, ~r/does not accept :image input/, fn ->
          Provider.stream(provider, [message_with([%{data: @png}])])
        end

      assert error.type == :unsupported
      assert error.provider == MinimalProvider
    end

    test "given a supported modality, then the stream runs" do
      sse_plug = fn conn ->
        body =
          ~s(data: {"candidates":[{"content":{"parts":[{"text":"seen"}]}}]}\n\n) <>
            "data: [DONE]\n\n"

        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, body)
      end

      provider = gemini_provider(sse_plug)

      assert ["seen"] =
               provider
               |> Provider.stream([message_with([%{data: @png}])])
               |> Enum.filter(&(&1.type == :text_delta))
               |> Enum.map(& &1.text)
    end
  end
end
