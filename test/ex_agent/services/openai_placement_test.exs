defmodule ExAgent.Services.OpenAIPlacementTest do
  @moduledoc """
  Covers how the OpenAI service decides to deliver an attachment: referenced by
  URL, inlined as a base64 data URI, or uploaded through the Files API.

  Size boundaries are driven by `Attachment.byte_size` rather than by real
  multi-megabyte payloads — that is the field the placement logic reads, so the
  production path runs without base64-encoding 20 MB per test.
  """

  # Shares the global ExAgent.UploadCache with the other cache users.
  use ExUnit.Case, async: false

  alias ExAgent.{Attachment, FileRef, Message, Source, UploadCache}
  alias ExAgent.Providers.OpenAI
  alias ExAgent.Services.OpenAIService

  @inline_limit 20_000_000
  @base_url "https://api.openai.com/v1"

  setup do
    UploadCache.clear()
    %{dir: tmp_dir()}
  end

  defp tmp_dir do
    dir = Path.join(System.tmp_dir!(), "ex_agent_oai_place_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf(dir) end)
    dir
  end

  defp chat_response(text) do
    %{"choices" => [%{"message" => %{"role" => "assistant", "content" => text}}]}
  end

  defp recording_provider(opts \\ []) do
    test_pid = self()
    file_id = Keyword.get(opts, :file_id, "file-abc")

    plug = fn conn ->
      if conn.request_path == "/files" do
        send(test_pid, {:upload, conn.request_path})
        Req.Test.json(conn, %{"id" => file_id, "filename" => "upload"})
      else
        {body, conn} = read_all(conn)
        send(test_pid, {:chat, Jason.decode!(body)})
        Req.Test.json(conn, chat_response("ok"))
      end
    end

    %OpenAI{
      api_key: "sk-test",
      model: "gpt-4o",
      base_url: @base_url,
      upload_cache: Keyword.get(opts, :upload_cache, true),
      req: Req.new(plug: plug)
    }
  end

  defp read_all(conn, acc \\ "") do
    case Plug.Conn.read_body(conn) do
      {:ok, chunk, conn} -> {acc <> chunk, conn}
      {:more, chunk, conn} -> read_all(conn, acc <> chunk)
    end
  end

  defp message_with(files) do
    {:ok, msg} = Message.new(role: :user, content: "Look", attachments: files)
    msg
  end

  defp sized_attachment(mime_type, byte_size, data \\ "payload") do
    %Attachment{
      kind: :data,
      data: data,
      mime_type: mime_type,
      modality: Source.modality(mime_type),
      byte_size: byte_size,
      filename: "asset.bin"
    }
  end

  defp first_part do
    assert_received {:chat, body}
    [part | _] = hd(body["messages"])["content"]
    part
  end

  defp upload_count(count \\ 0) do
    receive do
      {:upload, _path} -> upload_count(count + 1)
    after
      0 -> count
    end
  end

  describe "under the limit, attachments inline" do
    test "given a small image on disk, then it becomes a data URI", %{dir: dir} do
      path = Path.join(dir, "shot.png")
      File.write!(path, <<0x89, "PNG", 0x0D, 0x0A, 0x1A, 0x0A, 0::size(64)>>)

      assert {:ok, _} = OpenAIService.chat(recording_provider(), [message_with([%{path: path}])])

      assert %{"type" => "image_url", "image_url" => %{"url" => url}} = first_part()
      assert String.starts_with?(url, "data:image/png;base64,")
      assert upload_count() == 0
    end

    test "given a small document, then it is sent as inline file_data", %{dir: dir} do
      path = Path.join(dir, "report.pdf")
      File.write!(path, "%PDF-1.7 body")

      assert {:ok, _} = OpenAIService.chat(recording_provider(), [message_with([%{path: path}])])

      assert %{"type" => "file", "file" => file} = first_part()
      assert file["filename"] == "report.pdf"
      assert String.starts_with?(file["file_data"], "data:application/pdf;base64,")
      assert upload_count() == 0
    end

    test "given a file exactly at the limit, then it still inlines" do
      attachment = sized_attachment("image/png", @inline_limit)

      assert {:ok, _} = OpenAIService.chat(recording_provider(), [message_with([attachment])])

      assert %{"type" => "image_url"} = first_part()
      assert upload_count() == 0
    end
  end

  describe "over the limit, attachments upload" do
    # Chat completions has no content part that references an uploaded image, so
    # uploading one would spend the round trip and then guarantee a 400.
    test "given an image over the limit, then it is refused instead of uploaded" do
      provider = recording_provider(file_id: "file-large-image")
      attachment = sized_attachment("image/png", @inline_limit + 1)

      assert {:error, %ExAgent.Error{type: :unsupported} = error} =
               OpenAIService.chat(provider, [message_with([attachment])])

      assert error.message =~ "over OpenAI's"
      assert upload_count() == 0
      refute_received {:chat, _}
    end

    test "given a document over the limit, then it uploads and references the file id" do
      provider = recording_provider(file_id: "file-large-doc")
      attachment = sized_attachment("application/pdf", @inline_limit + 1)

      assert {:ok, _} = OpenAIService.chat(provider, [message_with([attachment])])

      assert upload_count() == 1
      assert %{"type" => "file", "file" => %{"file_id" => "file-large-doc"}} = first_part()
    end

    test "given the upload fails, then the error surfaces and no chat request is made" do
      provider = %OpenAI{
        api_key: "sk-test",
        model: "gpt-4o",
        base_url: @base_url,
        req:
          Req.new(
            plug: fn conn ->
              conn
              |> Plug.Conn.put_resp_content_type("application/json")
              |> Plug.Conn.send_resp(413, Jason.encode!(%{"error" => %{"message" => "too big"}}))
            end
          )
      }

      attachment = sized_attachment("application/pdf", @inline_limit + 1)

      assert {:error, %ExAgent.Error{status: 413} = error} =
               OpenAIService.chat(provider, [message_with([attachment])])

      assert error.message == "too big"
      refute_received {:chat, _}
    end
  end

  describe "references pass through untouched" do
    test "given an image url, then it is neither uploaded nor encoded" do
      files = [%{url: "https://cdn.example.com/huge.png"}]

      assert {:ok, _} = OpenAIService.chat(recording_provider(), [message_with(files)])

      assert %{"image_url" => %{"url" => "https://cdn.example.com/huge.png"}} = first_part()
      assert upload_count() == 0
    end

    test "given an existing file_ref, then it is referenced without re-uploading" do
      ref = %FileRef{provider: :openai, file_id: "file-already", mime_type: "application/pdf"}

      assert {:ok, _} =
               OpenAIService.chat(recording_provider(), [message_with([%{file_ref: ref}])])

      assert %{"file" => %{"file_id" => "file-already"}} = first_part()
      assert upload_count() == 0
    end
  end

  describe "upload reuse" do
    test "given the same bytes twice, then only one upload is issued" do
      provider = recording_provider()
      attachment = sized_attachment("application/pdf", @inline_limit + 1, "identical bytes")

      assert {:ok, _} = OpenAIService.chat(provider, [message_with([attachment])])
      assert {:ok, _} = OpenAIService.chat(provider, [message_with([attachment])])

      assert upload_count() == 1
    end

    test "given upload_cache: false, then every call re-uploads" do
      provider = recording_provider(upload_cache: false)
      attachment = sized_attachment("application/pdf", @inline_limit + 1, "identical bytes")

      assert {:ok, _} = OpenAIService.chat(provider, [message_with([attachment])])
      assert {:ok, _} = OpenAIService.chat(provider, [message_with([attachment])])

      assert upload_count() == 2
    end

    test "given the same bytes under a different api key, then it re-uploads" do
      attachment = sized_attachment("application/pdf", @inline_limit + 1, "shared bytes")

      assert {:ok, _} =
               OpenAIService.chat(recording_provider(), [message_with([attachment])])

      other = %{recording_provider() | api_key: "sk-other-account"}
      assert {:ok, _} = OpenAIService.chat(other, [message_with([attachment])])

      # A file id from one account is useless to another.
      assert upload_count() == 2
    end
  end

  describe "lazy path reads" do
    test "given a file deleted after normalization, then the error surfaces", %{dir: dir} do
      path = Path.join(dir, "vanishing.png")
      File.write!(path, "data")
      msg = message_with([%{path: path}])
      File.rm!(path)

      assert {:error, %ExAgent.Error{type: :invalid_request} = error} =
               OpenAIService.chat(recording_provider(), [msg])

      assert error.message =~ "failed to read file"
      refute_received {:chat, _}
      assert upload_count() == 0
    end
  end
end
