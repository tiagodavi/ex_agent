defmodule ExAgent.Services.GeminiPlacementTest do
  @moduledoc """
  Covers how the Gemini service decides to deliver an attachment: referenced by
  URL, inlined as base64, or uploaded through the Files API.

  Size boundaries are driven by `Attachment.byte_size` rather than by real
  multi-megabyte payloads. That is the field the placement logic actually reads,
  so the production path is exercised without base64-encoding 50 MB per test.
  """

  # Shares the global ExAgent.UploadCache with ExAgent.UploadCacheTest.
  use ExUnit.Case, async: false

  alias ExAgent.{Attachment, FileRef, Message, Source, UploadCache}
  alias ExAgent.Providers.Gemini
  alias ExAgent.Services.GeminiService

  # Gemini inlines up to 20 MB, or 50 MB for PDFs.
  @inline_limit 20_000_000
  @pdf_inline_limit 50_000_000

  @base_url "https://generativelanguage.googleapis.com/v1beta"

  setup do
    UploadCache.clear()
    %{dir: tmp_dir()}
  end

  defp tmp_dir do
    dir = Path.join(System.tmp_dir!(), "ex_agent_placement_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf(dir) end)
    dir
  end

  defp generate_response(text) do
    %{"candidates" => [%{"content" => %{"parts" => [%{"text" => text}], "role" => "model"}}]}
  end

  defp upload_response(uri) do
    %{
      "file" => %{
        "uri" => uri,
        "name" => "files/generated",
        "state" => "ACTIVE",
        "mimeType" => "video/mp4",
        "displayName" => "clip.mp4"
      }
    }
  end

  # Records each request to the test process so a test can assert both what was
  # sent and — via refute_received — what was not.
  defp recording_provider(opts \\ []) do
    test_pid = self()
    upload_uri = Keyword.get(opts, :upload_uri, "https://files.google/abc")

    plug = fn conn ->
      if String.contains?(conn.request_path, "/upload/") do
        send(test_pid, {:upload, conn.request_path})
        Req.Test.json(conn, upload_response(upload_uri))
      else
        {body, conn} = read_all(conn)
        send(test_pid, {:generate, Jason.decode!(body)})
        Req.Test.json(conn, generate_response("ok"))
      end
    end

    %Gemini{
      api_key: "AIza-test",
      model: "gemini-2.0-flash",
      base_url: @base_url,
      upload_cache: Keyword.get(opts, :upload_cache, true),
      req: Req.new(plug: plug)
    }
  end

  # read_body/2 returns {:more, _, _} once a body exceeds its read length.
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

  # An attachment whose declared size crosses a threshold while its payload
  # stays tiny.
  defp sized_attachment(mime_type, byte_size, data \\ "payload") do
    %Attachment{
      kind: :data,
      data: data,
      mime_type: mime_type,
      modality: Source.modality(mime_type),
      byte_size: byte_size
    }
  end

  defp first_part do
    assert_received {:generate, body}
    [part | _] = hd(body["contents"])["parts"]
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
    test "given a small file on disk, then it is inlined and nothing is uploaded", %{dir: dir} do
      path = Path.join(dir, "note.txt")
      File.write!(path, "hello world")

      assert {:ok, _} = GeminiService.chat(recording_provider(), [message_with([%{path: path}])])

      assert %{"inline_data" => %{"mime_type" => "text/plain", "data" => data}} = first_part()
      assert Base.decode64!(data) == "hello world"
      assert upload_count() == 0
    end

    test "given a file exactly at the limit, then it still inlines" do
      attachment = sized_attachment("image/png", @inline_limit)

      assert {:ok, _} = GeminiService.chat(recording_provider(), [message_with([attachment])])

      assert %{"inline_data" => _} = first_part()
      assert upload_count() == 0
    end

    test "given a PDF over the general limit but under the PDF limit, then it inlines" do
      attachment = sized_attachment("application/pdf", @inline_limit + 1)

      assert {:ok, _} = GeminiService.chat(recording_provider(), [message_with([attachment])])

      assert %{"inline_data" => %{"mime_type" => "application/pdf"}} = first_part()
      assert upload_count() == 0
    end
  end

  describe "over the limit, attachments upload" do
    test "given a file over the limit, then it uploads and is referenced by URI" do
      provider = recording_provider(upload_uri: "https://files.google/clip")
      attachment = sized_attachment("video/mp4", @inline_limit + 1)

      assert {:ok, _} = GeminiService.chat(provider, [message_with([attachment])])

      assert upload_count() == 1
      assert %{"file_data" => %{"file_uri" => "https://files.google/clip"}} = first_part()
    end

    test "given a PDF over the PDF limit, then it uploads" do
      attachment = sized_attachment("application/pdf", @pdf_inline_limit + 1)

      assert {:ok, _} = GeminiService.chat(recording_provider(), [message_with([attachment])])

      assert upload_count() == 1
      assert %{"file_data" => _} = first_part()
    end

    test "given the upload fails, then the error surfaces and no generate request is made" do
      provider = %Gemini{
        api_key: "AIza-test",
        model: "gemini-2.0-flash",
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

      attachment = sized_attachment("video/mp4", @inline_limit + 1)

      assert {:error, %ExAgent.Error{status: 413} = error} =
               GeminiService.chat(provider, [message_with([attachment])])

      assert error.message == "too big"
      refute_received {:generate, _}
    end
  end

  describe "video end to end" do
    # Task 4's acceptance criterion: a local video uploads, waits for the file to
    # reach ACTIVE, and is then referenced by URI in the generate request.
    test "given a local .mp4, then it uploads, waits for ACTIVE, and is referenced by URI",
         %{dir: dir} do
      test_pid = self()
      counter = :counters.new(1, [:atomics])

      plug = fn conn ->
        cond do
          String.contains?(conn.request_path, "/upload/") ->
            send(test_pid, {:upload, conn.request_path})

            # Large files and video land in PROCESSING, not ACTIVE.
            Req.Test.json(conn, %{
              "file" => %{
                "uri" => "https://files.google/clip",
                "name" => "files/clip",
                "state" => "PROCESSING",
                "mimeType" => "video/mp4"
              }
            })

          conn.method == "GET" ->
            :counters.add(counter, 1, 1)
            polls = :counters.get(counter, 1)
            send(test_pid, {:poll, polls})

            state = if polls >= 2, do: "ACTIVE", else: "PROCESSING"

            Req.Test.json(conn, %{
              "name" => "files/clip",
              "uri" => "https://files.google/clip",
              "state" => state,
              "mimeType" => "video/mp4"
            })

          true ->
            {body, conn} = read_all(conn)
            send(test_pid, {:generate, Jason.decode!(body)})
            Req.Test.json(conn, generate_response("A cat video"))
        end
      end

      provider = %Gemini{
        api_key: "AIza-test",
        model: "gemini-2.0-flash",
        base_url: @base_url,
        req: Req.new(plug: plug)
      }

      path = Path.join(dir, "clip.mp4")
      File.write!(path, "fake mp4 bytes")

      # Force the upload path without materializing 20 MB.
      {:ok, attachment} = Attachment.new(%{path: path})
      attachment = %{attachment | byte_size: @inline_limit + 1}

      assert {:ok, _} =
               GeminiService.chat(provider, [message_with([attachment])],
                 poll_interval_ms: 0,
                 max_poll_attempts: 5
               )

      assert_received {:upload, _}
      # It polled until the file left PROCESSING rather than referencing it early.
      assert_received {:poll, 1}
      assert_received {:poll, 2}

      assert %{"file_data" => %{"file_uri" => "https://files.google/clip"}} = first_part()
    end

    test "given a file stuck in PROCESSING, then it times out rather than looping forever",
         %{dir: dir} do
      plug = fn conn ->
        if String.contains?(conn.request_path, "/upload/") do
          Req.Test.json(conn, %{
            "file" => %{"uri" => "u", "name" => "files/x", "state" => "PROCESSING"}
          })
        else
          Req.Test.json(conn, %{"name" => "files/x", "state" => "PROCESSING"})
        end
      end

      provider = %Gemini{
        api_key: "AIza-test",
        model: "gemini-2.0-flash",
        base_url: @base_url,
        req: Req.new(plug: plug)
      }

      path = Path.join(dir, "stuck.mp4")
      File.write!(path, "bytes")
      {:ok, attachment} = Attachment.new(%{path: path})
      attachment = %{attachment | byte_size: @inline_limit + 1}

      assert {:error, %ExAgent.Error{type: :timeout} = error} =
               GeminiService.chat(provider, [message_with([attachment])],
                 poll_interval_ms: 0,
                 max_poll_attempts: 3
               )

      assert error.message =~ "ACTIVE"
      assert error.retryable?
    end
  end

  describe "references pass through untouched" do
    test "given a url attachment, then it is neither uploaded nor inlined" do
      files = [%{url: "https://cdn.example.com/huge.mp4"}]

      assert {:ok, _} = GeminiService.chat(recording_provider(), [message_with(files)])

      assert %{"file_data" => %{"file_uri" => "https://cdn.example.com/huge.mp4"}} = first_part()
      assert upload_count() == 0
    end

    test "given an existing file_ref, then it is referenced without re-uploading" do
      ref = %FileRef{provider: :gemini, file_uri: "files/already", mime_type: "video/mp4"}

      assert {:ok, _} =
               GeminiService.chat(recording_provider(), [message_with([%{file_ref: ref}])])

      assert %{"file_data" => %{"file_uri" => "files/already"}} = first_part()
      assert upload_count() == 0
    end
  end

  describe "upload reuse" do
    test "given the same bytes twice, then only one upload is issued" do
      provider = recording_provider(upload_uri: "https://files.google/once")
      attachment = sized_attachment("video/mp4", @inline_limit + 1, "identical bytes")

      assert {:ok, _} = GeminiService.chat(provider, [message_with([attachment])])
      assert {:ok, _} = GeminiService.chat(provider, [message_with([attachment])])

      assert upload_count() == 1
    end

    test "given different bytes, then each uploads" do
      provider = recording_provider()

      assert {:ok, _} =
               GeminiService.chat(provider, [
                 message_with([sized_attachment("video/mp4", @inline_limit + 1, "first")])
               ])

      assert {:ok, _} =
               GeminiService.chat(provider, [
                 message_with([sized_attachment("video/mp4", @inline_limit + 1, "second")])
               ])

      assert upload_count() == 2
    end

    test "given upload_cache: false, then every call re-uploads" do
      provider = recording_provider(upload_cache: false)
      attachment = sized_attachment("video/mp4", @inline_limit + 1, "identical bytes")

      assert {:ok, _} = GeminiService.chat(provider, [message_with([attachment])])
      assert {:ok, _} = GeminiService.chat(provider, [message_with([attachment])])

      assert upload_count() == 2
    end

    test "given an expired cached ref, then it re-uploads rather than reusing it" do
      provider = recording_provider(upload_uri: "https://files.google/fresh")
      data = "stale payload"
      scope = UploadCache.scope(Gemini, @base_url, "AIza-test")

      UploadCache.put(scope, data, %FileRef{
        provider: :gemini,
        file_uri: "files/stale",
        mime_type: "video/mp4",
        expires_at: DateTime.add(DateTime.utc_now(), -60, :second)
      })

      attachment = sized_attachment("video/mp4", @inline_limit + 1, data)
      assert {:ok, _} = GeminiService.chat(provider, [message_with([attachment])])

      assert upload_count() == 1
      assert %{"file_data" => %{"file_uri" => "https://files.google/fresh"}} = first_part()
    end
  end

  describe "video metadata" do
    test "given fps, then it is sent alongside the media part" do
      files = [%{url: "https://cdn.example.com/clip.mp4", fps: 2}]

      assert {:ok, _} = GeminiService.chat(recording_provider(), [message_with(files)])

      assert %{"video_metadata" => %{"fps" => 2}} = first_part()
    end

    test "given string provider_opts, then they pass through verbatim" do
      files = [
        %{url: "https://cdn.example.com/clip.mp4", provider_opts: %{"start_offset" => "10s"}}
      ]

      assert {:ok, _} = GeminiService.chat(recording_provider(), [message_with(files)])

      assert %{"video_metadata" => %{"start_offset" => "10s"}} = first_part()
    end

    test "given no video options, then no video_metadata key is emitted" do
      files = [%{url: "https://cdn.example.com/clip.mp4"}]

      assert {:ok, _} = GeminiService.chat(recording_provider(), [message_with(files)])

      refute Map.has_key?(first_part(), "video_metadata")
    end
  end

  describe "lazy path reads" do
    test "given a file deleted after normalization, then the error surfaces", %{dir: dir} do
      path = Path.join(dir, "vanishing.png")
      File.write!(path, "data")
      msg = message_with([%{path: path}])
      File.rm!(path)

      assert {:error, %ExAgent.Error{type: :invalid_request} = error} =
               GeminiService.chat(recording_provider(), [msg])

      assert error.message =~ "failed to read file"
      refute_received {:generate, _}
      assert upload_count() == 0
    end
  end
end
