defmodule ExAgent.Services.GeminiUploadServiceTest do
  use ExUnit.Case, async: true

  alias ExAgent.Services.GeminiUploadService
  alias ExAgent.FileRef

  defp upload_success_response(name, uri, opts \\ []) do
    file =
      %{
        "name" => name,
        "displayName" => opts[:display_name] || "upload",
        "mimeType" => opts[:mime_type] || "application/pdf",
        "sizeBytes" => "12345",
        "uri" => uri,
        "state" => opts[:state] || "ACTIVE"
      }

    file =
      if Keyword.has_key?(opts, :expiration),
        do: Map.put(file, "expirationTime", opts[:expiration]),
        else: Map.put(file, "expirationTime", "2026-03-24T12:00:00Z")

    %{"file" => file}
  end

  # Happy path tests
  describe "upload/4 success" do
    test "uploads a file and returns a FileRef with file_uri" do
      req =
        Req.new(
          plug: fn conn ->
            Req.Test.json(
              conn,
              upload_success_response(
                "files/abc123",
                "https://generativelanguage.googleapis.com/v1beta/files/abc123"
              )
            )
          end
        )

      assert {:ok,
              %FileRef{
                provider: :gemini,
                file_uri: "https://generativelanguage.googleapis.com/v1beta/files/abc123",
                mime_type: "application/pdf"
              }} =
               GeminiUploadService.upload("AIza-test", "pdf data", "application/pdf",
                 filename: "report.pdf",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files"
               )
    end

    test "sends multipart/related with correct headers" do
      req =
        Req.new(
          plug: fn conn ->
            [content_type] = Plug.Conn.get_req_header(conn, "content-type")
            assert String.starts_with?(content_type, "multipart/related; boundary=")
            [protocol] = Plug.Conn.get_req_header(conn, "x-goog-upload-protocol")
            assert protocol == "multipart"
            [api_key] = Plug.Conn.get_req_header(conn, "x-goog-api-key")
            assert api_key == "AIza-test-key"

            Req.Test.json(
              conn,
              upload_success_response("files/xyz", "https://example.com/files/xyz")
            )
          end
        )

      assert {:ok, _ref} =
               GeminiUploadService.upload("AIza-test-key", "data", "image/png",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files"
               )
    end

    test "parses expiration time from response" do
      req =
        Req.new(
          plug: fn conn ->
            Req.Test.json(
              conn,
              upload_success_response("files/exp", "https://example.com/files/exp",
                expiration: "2026-03-24T12:00:00Z"
              )
            )
          end
        )

      assert {:ok, %FileRef{expires_at: %DateTime{year: 2026, month: 3, day: 24}}} =
               GeminiUploadService.upload("AIza-test", "data", "text/plain",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files"
               )
    end
  end

  # Bad path tests
  describe "upload/4 errors" do
    test "returns error for non-200 status" do
      req =
        Req.new(
          plug: fn conn ->
            conn |> Plug.Conn.send_resp(403, Jason.encode!(%{"error" => "forbidden"}))
          end
        )

      assert {:error, {403, _}} =
               GeminiUploadService.upload("AIza-test", "data", "text/plain",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files"
               )
    end

    test "returns error for server errors" do
      req =
        Req.new(
          plug: fn conn ->
            conn |> Plug.Conn.send_resp(500, Jason.encode!(%{"error" => "internal"}))
          end
        )

      assert {:error, {500, _}} =
               GeminiUploadService.upload("AIza-test", "data", "text/plain",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files"
               )
    end

    test "returns error when file processing fails" do
      counter = :counters.new(1, [:atomics])

      req =
        Req.new(
          plug: fn conn ->
            :counters.add(counter, 1, 1)
            count = :counters.get(counter, 1)

            if count == 1 do
              Req.Test.json(
                conn,
                upload_success_response("files/fail", "https://example.com/files/fail",
                  state: "PROCESSING"
                )
              )
            else
              Req.Test.json(conn, %{
                "name" => "files/fail",
                "state" => "FAILED",
                "mimeType" => "text/plain"
              })
            end
          end
        )

      assert {:error, :file_processing_failed} =
               GeminiUploadService.upload("AIza-test", "data", "text/plain",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files",
                 poll_base_url: "http://localhost/v1beta"
               )
    end
  end

  # Edge case tests
  describe "upload/4 edge cases" do
    test "defaults filename to 'upload'" do
      req =
        Req.new(
          plug: fn conn ->
            {:ok, body, conn} = Plug.Conn.read_body(conn)
            assert body =~ "\"display_name\":\"upload\""

            Req.Test.json(
              conn,
              upload_success_response("files/def", "https://example.com/files/def")
            )
          end
        )

      assert {:ok, %FileRef{filename: "upload"}} =
               GeminiUploadService.upload("AIza-test", "data", "application/octet-stream",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files"
               )
    end

    test "handles nil expiration time" do
      req =
        Req.new(
          plug: fn conn ->
            Req.Test.json(
              conn,
              upload_success_response("files/noexp", "https://example.com/files/noexp",
                expiration: nil
              )
            )
          end
        )

      assert {:ok, %FileRef{expires_at: nil}} =
               GeminiUploadService.upload("AIza-test", "data", "text/plain",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files"
               )
    end

    test "returns ok immediately when state is ACTIVE" do
      req =
        Req.new(
          plug: fn conn ->
            Req.Test.json(
              conn,
              upload_success_response("files/active", "https://example.com/files/active",
                state: "ACTIVE"
              )
            )
          end
        )

      assert {:ok, %FileRef{file_uri: "https://example.com/files/active"}} =
               GeminiUploadService.upload("AIza-test", "data", "text/plain",
                 req: req,
                 upload_url: "http://localhost/upload/v1beta/files"
               )
    end
  end
end
