defmodule ExAgent.Services.OpenAIUploadServiceTest do
  use ExUnit.Case, async: true

  alias ExAgent.Services.OpenAIUploadService
  alias ExAgent.FileRef

  defp build_req(plug_fn) do
    Req.new(plug: plug_fn)
  end

  defp upload_success_response(file_id, filename) do
    %{
      "id" => file_id,
      "object" => "file",
      "bytes" => 12345,
      "created_at" => 1_700_000_000,
      "filename" => filename,
      "purpose" => "user_data"
    }
  end

  # Happy path tests
  describe "upload/4 success" do
    test "uploads a file and returns a FileRef with file_id" do
      req =
        build_req(fn conn ->
          Req.Test.json(conn, upload_success_response("file-abc123", "report.pdf"))
        end)

      assert {:ok,
              %FileRef{provider: :openai, file_id: "file-abc123", mime_type: "application/pdf"}} =
               OpenAIUploadService.upload(req, "pdf binary data", "application/pdf",
                 filename: "report.pdf"
               )
    end

    test "sends multipart/form-data with correct boundary" do
      req =
        build_req(fn conn ->
          [content_type] = Plug.Conn.get_req_header(conn, "content-type")
          assert String.starts_with?(content_type, "multipart/form-data; boundary=")
          Req.Test.json(conn, upload_success_response("file-xyz", "image.png"))
        end)

      assert {:ok, _ref} =
               OpenAIUploadService.upload(req, "png data", "image/png", filename: "image.png")
    end

    test "includes purpose and file fields in multipart body" do
      req =
        build_req(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          assert body =~ "name=\"purpose\""
          assert body =~ "user_data"
          assert body =~ "name=\"file\""
          assert body =~ "filename=\"doc.txt\""
          assert body =~ "Content-Type: text/plain"
          Req.Test.json(conn, upload_success_response("file-txt", "doc.txt"))
        end)

      assert {:ok, _ref} =
               OpenAIUploadService.upload(req, "hello world", "text/plain", filename: "doc.txt")
    end
  end

  # Bad path tests
  describe "upload/4 errors" do
    test "returns error for non-200 status" do
      req =
        build_req(fn conn ->
          conn |> Plug.Conn.send_resp(401, Jason.encode!(%{"error" => "unauthorized"}))
        end)

      assert {:error, {401, _}} =
               OpenAIUploadService.upload(req, "data", "text/plain")
    end

    test "returns error for rate limiting" do
      req =
        build_req(fn conn ->
          conn |> Plug.Conn.send_resp(429, Jason.encode!(%{"error" => "rate_limited"}))
        end)

      assert {:error, {429, _}} =
               OpenAIUploadService.upload(req, "data", "text/plain")
    end

    test "returns error for server errors" do
      req =
        build_req(fn conn ->
          conn |> Plug.Conn.send_resp(500, Jason.encode!(%{"error" => "internal"}))
        end)

      assert {:error, {500, _}} =
               OpenAIUploadService.upload(req, "data", "text/plain")
    end
  end

  # Edge case tests
  describe "upload/4 edge cases" do
    test "defaults filename to 'upload' when not provided" do
      req =
        build_req(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          assert body =~ "filename=\"upload\""
          Req.Test.json(conn, upload_success_response("file-default", "upload"))
        end)

      assert {:ok, %FileRef{filename: "upload"}} =
               OpenAIUploadService.upload(req, "data", "application/octet-stream")
    end

    test "allows custom purpose" do
      req =
        build_req(fn conn ->
          {:ok, body, conn} = Plug.Conn.read_body(conn)
          assert body =~ "assistants"
          Req.Test.json(conn, upload_success_response("file-assist", "data.json"))
        end)

      assert {:ok, _ref} =
               OpenAIUploadService.upload(req, "data", "application/json",
                 purpose: "assistants",
                 filename: "data.json"
               )
    end

    test "uses filename from response when available" do
      req =
        build_req(fn conn ->
          Req.Test.json(conn, upload_success_response("file-renamed", "server_name.pdf"))
        end)

      assert {:ok, %FileRef{filename: "server_name.pdf"}} =
               OpenAIUploadService.upload(req, "data", "application/pdf", filename: "local.pdf")
    end
  end
end
