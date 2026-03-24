defmodule ExAgent.Services.GeminiUploadService do
  @moduledoc """
  HTTP service for uploading files to the Gemini Files API.

  Handles multipart/related uploads to the Gemini media upload endpoint
  and returns an `ExAgent.FileRef` with the provider-assigned file URI.
  Files uploaded to Gemini expire after 48 hours.
  """

  alias ExAgent.FileRef

  @upload_base_url "https://generativelanguage.googleapis.com/upload/v1beta/files"

  @max_poll_attempts 10
  @poll_interval_ms 1_000

  @doc """
  Uploads a file to Gemini and returns a file reference.

  Uses `multipart/related` with JSON metadata and binary content.
  Polls for file processing completion if the file state is `PROCESSING`.

  ## Options

  - `:filename` - display name for the file (default: `"upload"`)
  - `:upload_url` - override upload URL (useful for testing)
  """
  @spec upload(String.t(), binary(), String.t(), keyword()) ::
          {:ok, FileRef.t()} | {:error, term()}
  def upload(api_key, file_data, mime_type, opts \\ []) do
    filename = Keyword.get(opts, :filename, "upload")
    upload_url = Keyword.get(opts, :upload_url, @upload_base_url)

    boundary = "exagent-#{Base.encode16(:crypto.strong_rand_bytes(16), case: :lower)}"
    body = build_multipart_related(boundary, filename, file_data, mime_type)

    headers = [
      {"content-type", "multipart/related; boundary=#{boundary}"},
      {"x-goog-upload-protocol", "multipart"},
      {"x-goog-api-key", api_key}
    ]

    req = Keyword.get(opts, :req, Req.new())

    case Req.post(req, url: upload_url, headers: headers, body: body) do
      {:ok, %Req.Response{status: 200, body: %{"file" => file_info}}} ->
        ref = build_file_ref(file_info, mime_type, filename)
        maybe_wait_for_active(ref, api_key, file_info, opts)

      {:ok, %Req.Response{status: status, body: resp_body}} ->
        {:error, {status, resp_body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec build_file_ref(map(), String.t(), String.t()) :: FileRef.t()
  defp build_file_ref(file_info, mime_type, filename) do
    expires_at = parse_expiration(file_info["expirationTime"])

    %FileRef{
      provider: :gemini,
      file_uri: file_info["uri"],
      mime_type: file_info["mimeType"] || mime_type,
      filename: file_info["displayName"] || filename,
      expires_at: expires_at
    }
  end

  @spec maybe_wait_for_active(FileRef.t(), String.t(), map(), keyword()) ::
          {:ok, FileRef.t()} | {:error, term()}
  defp maybe_wait_for_active(ref, _api_key, %{"state" => "ACTIVE"}, _opts), do: {:ok, ref}

  defp maybe_wait_for_active(ref, api_key, %{"state" => "PROCESSING", "name" => name}, opts) do
    poll_until_active(ref, api_key, name, 0, opts)
  end

  defp maybe_wait_for_active(ref, _api_key, _file_info, _opts), do: {:ok, ref}

  @spec poll_until_active(FileRef.t(), String.t(), String.t(), non_neg_integer(), keyword()) ::
          {:ok, FileRef.t()} | {:error, term()}
  defp poll_until_active(_ref, _api_key, _name, attempt, _opts)
       when attempt >= @max_poll_attempts do
    {:error, :file_processing_timeout}
  end

  defp poll_until_active(ref, api_key, name, attempt, opts) do
    Process.sleep(@poll_interval_ms)

    base_url =
      Keyword.get(opts, :poll_base_url, "https://generativelanguage.googleapis.com/v1beta")

    req = Keyword.get(opts, :req, Req.new())

    case Req.get(req,
           url: "#{base_url}/#{name}",
           headers: [{"x-goog-api-key", api_key}]
         ) do
      {:ok, %Req.Response{status: 200, body: %{"state" => "ACTIVE"} = file_info}} ->
        {:ok, build_file_ref(file_info, ref.mime_type, ref.filename)}

      {:ok, %Req.Response{status: 200, body: %{"state" => "PROCESSING"}}} ->
        poll_until_active(ref, api_key, name, attempt + 1, opts)

      {:ok, %Req.Response{status: 200, body: %{"state" => "FAILED"}}} ->
        {:error, :file_processing_failed}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, {status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec build_multipart_related(String.t(), String.t(), binary(), String.t()) :: binary()
  defp build_multipart_related(boundary, filename, file_data, mime_type) do
    metadata = Jason.encode!(%{"file" => %{"display_name" => filename}})

    [
      "--#{boundary}\r\n",
      "Content-Type: application/json; charset=UTF-8\r\n\r\n",
      metadata,
      "\r\n",
      "--#{boundary}\r\n",
      "Content-Type: #{mime_type}\r\n\r\n",
      file_data,
      "\r\n",
      "--#{boundary}--\r\n"
    ]
    |> IO.iodata_to_binary()
  end

  @spec parse_expiration(String.t() | nil) :: DateTime.t() | nil
  defp parse_expiration(nil), do: nil

  defp parse_expiration(iso_string) when is_binary(iso_string) do
    case DateTime.from_iso8601(iso_string) do
      {:ok, dt, _offset} -> dt
      _ -> nil
    end
  end
end
