defmodule ExAgent.AttachmentTest do
  use ExUnit.Case, async: true

  alias ExAgent.{Attachment, FileRef}

  @png <<0x89, "PNG", 0x0D, 0x0A, 0x1A, 0x0A, 0::size(64)>>

  setup do
    dir =
      System.tmp_dir!() |> Path.join("ex_agent_attachment_#{System.unique_integer([:positive])}")

    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf(dir) end)
    %{dir: dir}
  end

  describe "new/1 happy paths" do
    test "given raw data with an explicit mime type, then it normalizes to a :data attachment" do
      assert {:ok, attachment} = Attachment.new(%{data: @png, mime_type: "image/png"})

      assert attachment.kind == :data
      assert attachment.mime_type == "image/png"
      assert attachment.modality == :image
      assert attachment.data == @png
      assert attachment.byte_size == byte_size(@png)
    end

    test "given a path, then the size is recorded but the bytes are not read yet", %{dir: dir} do
      path = Path.join(dir, "report.pdf")
      File.write!(path, "%PDF-1.7 body")

      assert {:ok, attachment} = Attachment.new(%{path: path})

      assert attachment.kind == :path
      assert attachment.path == path
      assert attachment.mime_type == "application/pdf"
      assert attachment.modality == :document
      assert attachment.filename == "report.pdf"
      assert attachment.byte_size == 13

      # Bytes stay on disk so a large file is not carried in conversation history.
      assert attachment.data == nil
    end

    test "given a url, then it is kept as a reference and never fetched" do
      assert {:ok, attachment} = Attachment.new(%{url: "https://cdn.example.com/invoice.png"})

      assert attachment.kind == :url
      assert attachment.url == "https://cdn.example.com/invoice.png"
      assert attachment.mime_type == "image/png"
      assert attachment.filename == "invoice.png"
      assert attachment.data == nil
      assert attachment.byte_size == nil
    end

    test "given a file_ref, then its mime type and filename carry over" do
      ref = %FileRef{
        provider: :openai,
        file_id: "file-1",
        mime_type: "application/pdf",
        filename: "a.pdf"
      }

      assert {:ok, attachment} = Attachment.new(%{file_ref: ref})

      assert attachment.kind == :file_ref
      assert attachment.file_ref == ref
      assert attachment.mime_type == "application/pdf"
      assert attachment.filename == "a.pdf"
    end
  end

  describe "new/1 mime inference" do
    test "given data with no mime type, then it is inferred from magic bytes" do
      assert {:ok, %{mime_type: "image/png"}} = Attachment.new(%{data: @png})
    end

    test "given a url with a query string, then the mime type still resolves" do
      assert {:ok, %{mime_type: "image/png"}} =
               Attachment.new(%{url: "https://cdn.example.com/a.png?sig=abc"})
    end

    test "given an explicit mime type, then it overrides inference" do
      assert {:ok, %{mime_type: "application/octet-stream"}} =
               Attachment.new(%{data: @png, mime_type: "application/octet-stream"})
    end
  end

  describe "new/1 provider options" do
    test "given fps and max_frames, then they are lifted into provider_opts" do
      assert {:ok, attachment} =
               Attachment.new(%{url: "https://example.com/clip.mp4", fps: 2, max_frames: 60})

      assert attachment.modality == :video
      assert attachment.provider_opts == %{fps: 2, max_frames: 60}
    end

    test "given an explicit provider_opts map, then unknown keys pass through" do
      assert {:ok, attachment} =
               Attachment.new(%{
                 url: "https://example.com/clip.mp4",
                 provider_opts: %{"videoMetadata" => %{"startOffset" => "10s"}}
               })

      assert attachment.provider_opts == %{"videoMetadata" => %{"startOffset" => "10s"}}
    end

    test "given no provider options, then provider_opts is empty" do
      assert {:ok, %{provider_opts: %{}}} = Attachment.new(%{data: @png})
    end
  end

  describe "new/1 failure paths" do
    test "given no source key, then it returns an actionable error" do
      assert {:error, message} = Attachment.new(%{mime_type: "image/png"})
      assert message =~ ":data, :path, :url, or :file_ref"
    end

    test "given an unreadable path, then it returns a read error", %{dir: dir} do
      assert {:error, message} = Attachment.new(%{path: Path.join(dir, "missing.png")})
      assert message =~ "failed to read file"
    end

    test "given a path with an unknown extension, then it asks for :mime_type", %{dir: dir} do
      path = Path.join(dir, "blob.qqq")
      File.write!(path, "x")

      assert {:error, message} = Attachment.new(%{path: path})
      assert message =~ ":mime_type"
    end

    test "given undetectable data, then it asks for :mime_type" do
      assert {:error, message} = Attachment.new(%{data: "plain text bytes"})
      assert message =~ ":mime_type"
    end
  end

  describe "bytes/1 and load/1" do
    test "given a path attachment, then bytes reads from disk on demand", %{dir: dir} do
      path = Path.join(dir, "doc.pdf")
      File.write!(path, "%PDF-1.7 body")
      {:ok, attachment} = Attachment.new(%{path: path})

      assert {:ok, "%PDF-1.7 body"} = Attachment.bytes(attachment)
    end

    test "given a path attachment, then load populates data", %{dir: dir} do
      path = Path.join(dir, "doc.pdf")
      File.write!(path, "%PDF-1.7 body")
      {:ok, attachment} = Attachment.new(%{path: path})

      assert {:ok, loaded} = Attachment.load(attachment)
      assert loaded.data == "%PDF-1.7 body"
      assert loaded.byte_size == 13
    end

    test "given a data attachment, then load is a no-op" do
      {:ok, attachment} = Attachment.new(%{data: @png})

      assert {:ok, ^attachment} = Attachment.load(attachment)
    end

    test "given a url attachment, then load passes through without fetching" do
      {:ok, attachment} = Attachment.new(%{url: "https://cdn.example.com/a.png"})

      assert {:ok, ^attachment} = Attachment.load(attachment)
      assert {:error, %ExAgent.Error{type: :invalid_request}} = Attachment.bytes(attachment)
    end

    test "given a file deleted after normalization, then bytes reports the failure", %{dir: dir} do
      path = Path.join(dir, "temp.pdf")
      File.write!(path, "body")
      {:ok, attachment} = Attachment.new(%{path: path})
      File.rm!(path)

      assert {:error, %ExAgent.Error{type: :invalid_request} = error} =
               Attachment.bytes(attachment)

      assert error.message =~ "failed to read file"
    end
  end

  describe "map compatibility with provider services" do
    test "given a data attachment, then it still matches the legacy data pattern" do
      {:ok, attachment} = Attachment.new(%{data: @png, mime_type: "image/png"})

      assert %{data: data, mime_type: "image/png"} = attachment
      assert data == @png
    end

    test "given a file_ref attachment, then it still matches the legacy file_ref pattern" do
      ref = %FileRef{provider: :gemini, file_uri: "files/x", mime_type: "image/png"}
      {:ok, attachment} = Attachment.new(%{file_ref: ref})

      assert %{file_ref: %FileRef{provider: :gemini}} = attachment
    end
  end
end
