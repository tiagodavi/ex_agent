defmodule ExAgent.Providers.OpenAICompatibleTest do
  use ExUnit.Case, async: true

  alias ExAgent.Providers.OpenAICompatible
  alias ExAgent.{Error, Message, Provider}

  @png <<0x89, "PNG", 0x0D, 0x0A, 0x1A, 0x0A, 0::size(64)>>

  defp chat_response(text) do
    %{"choices" => [%{"message" => %{"role" => "assistant", "content" => text}}]}
  end

  defp recording_provider(opts \\ []) do
    test_pid = self()

    plug = fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, conn.request_path, conn.req_headers, Jason.decode!(body)})
      Req.Test.json(conn, chat_response("ok"))
    end

    struct!(
      OpenAICompatible,
      Keyword.merge(
        [
          base_url: "https://modal.test/v1",
          model: "Qwen/Qwen3-VL-8B-Instruct",
          modalities: [:text, :image, :video],
          req: Req.new(plug: plug)
        ],
        opts
      )
    )
  end

  defp message_with(files) do
    {:ok, msg} = Message.new(role: :user, content: "Look", attachments: files)
    msg
  end

  defp sent do
    assert_received {:request, path, headers, body}
    %{path: path, headers: headers, body: body}
  end

  defp first_part(body), do: hd(hd(body["messages"])["content"])

  describe "new/1" do
    test "given required options, then it builds a provider with defaults" do
      provider = OpenAICompatible.new(base_url: "https://x.test/v1", model: "llama")

      assert provider.modalities == [:text]
      assert provider.max_inline_bytes == 33_554_432
      assert %Req.Request{} = provider.req
    end

    test "given a missing base_url, then it raises" do
      assert_raise NimbleOptions.ValidationError, fn ->
        OpenAICompatible.new(model: "llama")
      end
    end

    test "given a missing model, then it raises" do
      assert_raise NimbleOptions.ValidationError, fn ->
        OpenAICompatible.new(base_url: "https://x.test/v1")
      end
    end

    test "given no api_key, then no authorization header is set" do
      provider = OpenAICompatible.new(base_url: "https://x.test/v1", model: "llama")

      refute Enum.any?(provider.req.headers, fn {name, _} -> name == "authorization" end)
    end
  end

  describe "custom headers" do
    test "given Modal-style headers, then they reach the endpoint" do
      provider =
        recording_provider(
          headers: [{"Modal-Key", "mk-123"}, {"Modal-Secret", "ms-456"}],
          req: nil
        )

      # Rebuild through new/1 so header assembly is exercised, then swap in the stub.
      built =
        OpenAICompatible.new(
          base_url: "https://modal.test/v1",
          model: provider.model,
          headers: [{"Modal-Key", "mk-123"}, {"Modal-Secret", "ms-456"}]
        )

      test_pid = self()

      plug = fn conn ->
        send(test_pid, {:headers, conn.req_headers})
        Req.Test.json(conn, chat_response("ok"))
      end

      provider = %{built | req: Req.merge(built.req, plug: plug)}

      {:ok, msg} = Message.new(role: :user, content: "Hi")
      assert {:ok, _} = Provider.chat(provider, [msg])

      assert_received {:headers, headers}
      assert {"modal-key", "mk-123"} in headers
      assert {"modal-secret", "ms-456"} in headers
    end

    test "given an api_key, then it becomes a bearer header" do
      provider =
        OpenAICompatible.new(
          base_url: "https://x.test/v1",
          model: "llama",
          api_key: "sk-abc"
        )

      assert {"authorization", ["Bearer sk-abc"]} in provider.req.headers
    end

    test "given both api_key and an explicit authorization header, then the header wins" do
      provider =
        OpenAICompatible.new(
          base_url: "https://x.test/v1",
          model: "llama",
          api_key: "sk-abc",
          headers: [{"authorization", "Custom xyz"}]
        )

      assert {"authorization", ["Custom xyz"]} in provider.req.headers
    end
  end

  describe "chat/3" do
    test "given a text message, then it posts to /chat/completions" do
      {:ok, msg} = Message.new(role: :user, content: "Hi")

      assert {:ok, %ExAgent.Response{content: "ok"}} = Provider.chat(recording_provider(), [msg])

      assert %{path: "/chat/completions", body: body} = sent()
      assert body["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    end

    test "given raw image bytes, then they become a data URI in an image_url part" do
      files = [%{data: @png, mime_type: "image/png"}]

      assert {:ok, _} = Provider.chat(recording_provider(), [message_with(files)])

      part = first_part(sent().body)
      assert part["type"] == "image_url"
      assert String.starts_with?(part["image_url"]["url"], "data:image/png;base64,")
    end

    test "given a video url, then it becomes a video_url part passed through as-is" do
      files = [%{url: "https://cdn.example.com/clip.mp4"}]

      assert {:ok, _} = Provider.chat(recording_provider(), [message_with(files)])

      part = first_part(sent().body)
      assert part["type"] == "video_url"
      assert part["video_url"]["url"] == "https://cdn.example.com/clip.mp4"
    end

    test "given an audio url on a provider declaring audio, then it becomes an audio_url part" do
      provider = recording_provider(modalities: [:text, :audio])
      files = [%{url: "https://cdn.example.com/take.mp3"}]

      assert {:ok, _} = Provider.chat(provider, [message_with(files)])

      part = first_part(sent().body)
      assert part["type"] == "audio_url"
    end

    test "given fps and max_frames, then they are merged into the content part" do
      files = [%{url: "https://cdn.example.com/clip.mp4", fps: 2, max_frames: 60}]

      assert {:ok, _} = Provider.chat(recording_provider(), [message_with(files)])

      part = first_part(sent().body)
      assert part["fps"] == 2
      assert part["max_frames"] == 60
    end
  end

  describe "modality gating" do
    test "given video on a provider declaring only text and image, then it errors" do
      provider = recording_provider(modalities: [:text, :image])
      files = [%{url: "https://cdn.example.com/clip.mp4"}]

      assert {:error, %Error{type: :unsupported} = error} =
               Provider.chat(provider, [message_with(files)])

      assert error.message =~ "video"
      refute_received {:request, _, _, _}
    end

    test "given the default modalities, then even an image is rejected" do
      provider = recording_provider(modalities: [:text])

      assert {:error, %Error{type: :unsupported}} =
               Provider.chat(provider, [message_with([%{data: @png}])])
    end

    test "given a document, then it is rejected unless explicitly declared" do
      provider = recording_provider(modalities: [:text, :image, :video])
      files = [%{url: "https://cdn.example.com/report.pdf"}]

      assert {:error, %Error{type: :unsupported} = error} =
               Provider.chat(provider, [message_with(files)])

      assert error.message =~ "document"
    end
  end

  describe "max_inline_bytes" do
    test "given an attachment over the limit, then it returns a clear unsupported error" do
      provider = recording_provider(max_inline_bytes: 1_000)

      attachment = %ExAgent.Attachment{
        kind: :data,
        data: "small payload",
        mime_type: "image/png",
        modality: :image,
        byte_size: 50_000_000,
        filename: "huge.png"
      }

      assert {:error, %Error{type: :unsupported} = error} =
               Provider.chat(provider, [message_with([attachment])])

      assert error.message =~ "huge.png"
      assert error.message =~ "max_inline_bytes"
      assert error.message =~ "no Files API"
      refute_received {:request, _, _, _}
    end

    test "given an attachment under the limit, then it is inlined" do
      provider = recording_provider(max_inline_bytes: 50_000_000)

      assert {:ok, _} =
               Provider.chat(provider, [message_with([%{data: @png, mime_type: "image/png"}])])

      assert first_part(sent().body)["type"] == "image_url"
    end

    test "given a url over the limit, then it is unaffected — nothing is inlined" do
      provider = recording_provider(max_inline_bytes: 1)
      files = [%{url: "https://cdn.example.com/huge.mp4"}]

      assert {:ok, _} = Provider.chat(provider, [message_with(files)])
    end

    test "given a file_ref, then it is rejected — there is no Files API" do
      ref = %ExAgent.FileRef{provider: :openai, file_id: "file-x", mime_type: "image/png"}

      assert {:error, %Error{type: :unsupported} = error} =
               Provider.chat(recording_provider(), [message_with([%{file_ref: ref}])])

      assert error.message =~ "no Files API"
    end
  end

  describe "probe/1" do
    defp probing_provider(response, status \\ 200) do
      OpenAICompatible.new(base_url: "https://modal.test/v1", model: "Qwen/Qwen3-VL-8B-Instruct")
      |> then(fn provider ->
        plug = fn conn ->
          conn
          |> Plug.Conn.put_resp_content_type("application/json")
          |> Plug.Conn.send_resp(status, Jason.encode!(response))
        end

        %{provider | req: Req.merge(provider.req, plug: plug)}
      end)
    end

    test "given the configured model is served, then it returns :ok" do
      provider = probing_provider(%{"data" => [%{"id" => "Qwen/Qwen3-VL-8B-Instruct"}]})

      assert :ok = OpenAICompatible.probe(provider)
    end

    test "given a different model is served, then it reports the mismatch" do
      provider = probing_provider(%{"data" => [%{"id" => "meta-llama/Llama-3.1-8B"}]})

      assert {:error, %Error{type: :not_found} = error} = OpenAICompatible.probe(provider)
      assert error.message =~ "Qwen/Qwen3-VL-8B-Instruct"
      assert error.message =~ "meta-llama/Llama-3.1-8B"
    end

    test "given the endpoint is unreachable, then the http error surfaces" do
      provider = probing_provider(%{"error" => %{"message" => "nope"}}, 503)

      assert {:error, %Error{type: :server, status: 503}} = OpenAICompatible.probe(provider)
    end
  end

  describe "supported_modalities/1" do
    test "given a configured provider, then the declared modalities are reported" do
      provider =
        OpenAICompatible.new(base_url: "https://x/v1", model: "m", modalities: [:text, :video])

      assert Provider.supported_modalities(provider) == [:text, :video]
    end

    test "given no declaration, then it defaults to text only" do
      provider = OpenAICompatible.new(base_url: "https://x/v1", model: "m")

      assert Provider.supported_modalities(provider) == [:text]
    end
  end
end
