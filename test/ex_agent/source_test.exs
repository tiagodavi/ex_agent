defmodule ExAgent.SourceTest do
  use ExUnit.Case, async: true

  alias ExAgent.Source

  # Minimal valid-enough headers; inference reads only the leading magic bytes.
  @png <<0x89, "PNG", 0x0D, 0x0A, 0x1A, 0x0A, 0::size(64)>>
  @jpeg <<0xFF, 0xD8, 0xFF, 0xE0, 0::size(64)>>
  @gif <<"GIF89a", 0::size(64)>>
  @webp <<"RIFF", 0::size(32), "WEBP", 0::size(64)>>
  @wav <<"RIFF", 0::size(32), "WAVE", 0::size(64)>>
  @mp3_id3 <<"ID3", 0::size(64)>>
  @mp3_sync <<0xFF, 0xFB, 0::size(64)>>
  @mp4 <<0::size(32), "ftypisom", 0::size(64)>>
  @quicktime <<0::size(32), "ftypqt  ", 0::size(64)>>
  @m4a <<0::size(32), "ftypM4A ", 0::size(64)>>
  @pdf <<"%PDF-1.7", 0::size(64)>>

  describe "infer_mime/1 from a path" do
    test "given a known extension, then the mime type is derived from it" do
      assert {:ok, "image/png"} = Source.infer_mime({:path, "/tmp/shot.png"})
      assert {:ok, "application/pdf"} = Source.infer_mime({:path, "/tmp/report.pdf"})
      assert {:ok, "text/csv"} = Source.infer_mime({:path, "data/rows.csv"})
    end

    test "given an uppercase extension, then it still resolves" do
      assert {:ok, "image/jpeg"} = Source.infer_mime({:path, "/tmp/PHOTO.JPG"})
    end

    test "given a media extension the :mime registry lacks, then the fallback resolves it" do
      assert {:ok, "audio/flac"} = Source.infer_mime({:path, "/tmp/take.flac"})
      assert {:ok, "audio/mp4"} = Source.infer_mime({:path, "/tmp/take.m4a"})
      assert {:ok, "audio/mp4"} = Source.infer_mime({:path, "/tmp/BOOK.M4B"})
    end

    test "given video extensions, then they resolve to the video modality" do
      for path <- ~w(/tmp/a.mp4 /tmp/a.mov /tmp/a.webm) do
        assert {:ok, mime_type} = Source.infer_mime({:path, path})
        assert Source.modality(mime_type) == :video
      end
    end

    test "given no extension, then it returns an actionable error" do
      assert {:error, message} = Source.infer_mime({:path, "/tmp/noextension"})
      assert message =~ ":mime_type"
    end

    test "given an unknown extension, then it returns an actionable error" do
      assert {:error, message} = Source.infer_mime({:path, "/tmp/archive.qqq"})
      assert message =~ ":mime_type"
    end
  end

  describe "infer_mime/1 from a URL" do
    test "given a plain URL, then the mime type comes from the path extension" do
      assert {:ok, "image/png"} = Source.infer_mime({:url, "https://cdn.example.com/a/b.png"})
    end

    test "given a URL with a query string, then the query is ignored" do
      assert {:ok, "image/png"} =
               Source.infer_mime({:url, "https://cdn.example.com/invoice.png?sig=abc&x=1"})
    end

    test "given a URL with a fragment, then the fragment is ignored" do
      assert {:ok, "application/pdf"} =
               Source.infer_mime({:url, "https://example.com/doc.pdf#page=3"})
    end

    test "given a URL with no extension, then it returns an actionable error" do
      assert {:error, message} = Source.infer_mime({:url, "https://example.com/download"})
      assert message =~ ":mime_type"
    end
  end

  describe "infer_mime/1 from binary data" do
    test "given image magic bytes, then the image type is detected" do
      assert {:ok, "image/png"} = Source.infer_mime({:data, @png})
      assert {:ok, "image/jpeg"} = Source.infer_mime({:data, @jpeg})
      assert {:ok, "image/gif"} = Source.infer_mime({:data, @gif})
      assert {:ok, "image/webp"} = Source.infer_mime({:data, @webp})
    end

    test "given audio magic bytes, then the audio type is detected" do
      assert {:ok, "audio/wav"} = Source.infer_mime({:data, @wav})
      assert {:ok, "audio/mpeg"} = Source.infer_mime({:data, @mp3_id3})
      assert {:ok, "audio/mpeg"} = Source.infer_mime({:data, @mp3_sync})
    end

    test "given an ftyp box, then the brand distinguishes video from audio" do
      assert {:ok, "video/mp4"} = Source.infer_mime({:data, @mp4})
      assert {:ok, "video/quicktime"} = Source.infer_mime({:data, @quicktime})
      assert {:ok, "audio/mp4"} = Source.infer_mime({:data, @m4a})
    end

    test "given a PDF header, then the document type is detected" do
      assert {:ok, "application/pdf"} = Source.infer_mime({:data, @pdf})
    end

    test "given unrecognised bytes, then it returns an actionable error" do
      assert {:error, message} = Source.infer_mime({:data, "just some text"})
      assert message =~ ":mime_type"
    end

    test "given data shorter than any signature, then it does not crash" do
      assert {:error, _} = Source.infer_mime({:data, <<0x89>>})
      assert {:error, _} = Source.infer_mime({:data, ""})
    end

    test "given RIFF bytes that are neither WEBP nor WAVE, then it returns an error" do
      assert {:error, _} =
               Source.infer_mime({:data, <<"RIFF", 0::size(32), "AVI ", 0::size(64)>>})
    end
  end

  describe "modality/1" do
    test "given an image mime type, then the modality is :image" do
      assert Source.modality("image/png") == :image
      assert Source.modality("image/jpeg") == :image
    end

    test "given video and audio mime types, then the modality matches" do
      assert Source.modality("video/mp4") == :video
      assert Source.modality("audio/mpeg") == :audio
    end

    test "given a text or application mime type, then the modality is :document" do
      assert Source.modality("application/pdf") == :document
      assert Source.modality("text/csv") == :document
      assert Source.modality("text/plain") == :document
    end
  end
end
