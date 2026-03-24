defmodule ExAgent.Services.OpenAIUploadService do
  @moduledoc """
  HTTP service for uploading files to the OpenAI Files API.

  Handles multipart file uploads to `POST /v1/files` and returns
  an `ExAgent.FileRef` with the provider-assigned file ID.
  """

  alias ExAgent.FileRef

  @doc """
  Uploads a file to OpenAI and returns a file reference.

  Uses `multipart/form-data` with `purpose: "user_data"` by default.

  ## Options

  - `:filename` - original filename (default: `"upload"`)
  - `:purpose` - OpenAI file purpose (default: `"user_data"`)
  """
  @spec upload(Req.Request.t(), binary(), String.t(), keyword()) ::
          {:ok, FileRef.t()} | {:error, term()}
  def upload(req, file_data, mime_type, opts \\ []) do
    filename = Keyword.get(opts, :filename, "upload")
    purpose = Keyword.get(opts, :purpose, "user_data")

    boundary = "exagent-#{Base.encode16(:crypto.strong_rand_bytes(16), case: :lower)}"

    body = build_multipart_body(boundary, purpose, filename, file_data, mime_type)

    case Req.post(req,
           url: "/files",
           headers: [{"content-type", "multipart/form-data; boundary=#{boundary}"}],
           body: body
         ) do
      {:ok, %Req.Response{status: 200, body: %{"id" => file_id} = resp_body}} ->
        {:ok,
         %FileRef{
           provider: :openai,
           file_id: file_id,
           mime_type: mime_type,
           filename: resp_body["filename"] || filename
         }}

      {:ok, %Req.Response{status: status, body: resp_body}} ->
        {:error, {status, resp_body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec build_multipart_body(String.t(), String.t(), String.t(), binary(), String.t()) :: binary()
  defp build_multipart_body(boundary, purpose, filename, file_data, mime_type) do
    [
      "--#{boundary}\r\n",
      "Content-Disposition: form-data; name=\"purpose\"\r\n\r\n",
      "#{purpose}\r\n",
      "--#{boundary}\r\n",
      "Content-Disposition: form-data; name=\"file\"; filename=\"#{filename}\"\r\n",
      "Content-Type: #{mime_type}\r\n\r\n",
      file_data,
      "\r\n",
      "--#{boundary}--\r\n"
    ]
    |> IO.iodata_to_binary()
  end
end
