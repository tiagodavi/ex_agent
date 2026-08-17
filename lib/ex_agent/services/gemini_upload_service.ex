defmodule ExAgent.Services.GeminiUploadService do
  @moduledoc """
  HTTP service for uploading files to the Gemini Files API.

  Handles multipart/related uploads to the Gemini media upload endpoint
  and returns an `ExAgent.FileRef` with the provider-assigned file URI.
  Files uploaded to Gemini expire after 48 hours.
  """

  alias ExAgent.Error
  alias ExAgent.FileRef
  alias ExAgent.Providers.Gemini

  @upload_base_url "https://generativelanguage.googleapis.com/upload/v1beta/files"

  # Large files and video sit in PROCESSING for a while; referencing them early
  # fails. ~2 minutes of polling by default.
  @max_poll_attempts 60
  @poll_interval_ms 2_000

  # Req's own default is far too short for a multi-megabyte body: an upload that
  # was still streaming would be reported as a timeout. Polling is left on Req's
  # default - a status GET that hangs should give up quickly and be retried by
  # the next poll.
  @default_receive_timeout :timer.minutes(5)

  @doc """
  Uploads a file to Gemini and returns a file reference.

  Uses `multipart/related` with JSON metadata and binary content.
  Polls for file processing completion if the file state is `PROCESSING`.

  ## Options

  - `:filename` - display name for the file (default: `"upload"`)
  - `:upload_url` - override upload URL (useful for testing)
  - `:poll_interval_ms` - delay between processing polls (default: `2_000`)
  - `:max_poll_attempts` - polls before giving up (default: `60`, i.e. ~2 minutes)
  - `:receive_timeout` - milliseconds to wait for the upload (default: 5 minutes)
  """
  @spec upload(String.t(), binary(), String.t(), keyword()) ::
          {:ok, FileRef.t()} | {:error, Error.t()}
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

    Req.post(req,
      url: upload_url,
      headers: headers,
      body: body,
      receive_timeout: Keyword.get(opts, :receive_timeout, @default_receive_timeout)
    )
    |> Error.from_result(Gemini)
    |> case do
      {:ok, %{"file" => file_info}} ->
        ref = build_file_ref(file_info, mime_type, filename)
        maybe_wait_for_active(ref, api_key, file_info, opts)

      {:ok, resp_body} ->
        {:error, Error.unexpected_response(resp_body, Gemini)}

      {:error, _error} = failure ->
        failure
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
          {:ok, FileRef.t()} | {:error, Error.t()}
  defp maybe_wait_for_active(ref, _api_key, %{"state" => "ACTIVE"}, _opts), do: {:ok, ref}

  defp maybe_wait_for_active(ref, api_key, %{"state" => "PROCESSING", "name" => name}, opts) do
    poll_until_active(ref, api_key, name, 0, opts)
  end

  defp maybe_wait_for_active(ref, _api_key, _file_info, _opts), do: {:ok, ref}

  @spec poll_until_active(FileRef.t(), String.t(), String.t(), non_neg_integer(), keyword()) ::
          {:ok, FileRef.t()} | {:error, Error.t()}
  defp poll_until_active(ref, api_key, name, attempt, opts) do
    max_attempts = Keyword.get(opts, :max_poll_attempts, @max_poll_attempts)

    if attempt >= max_attempts do
      {:error,
       Error.new(:timeout, "file did not reach ACTIVE after #{max_attempts} polls", Gemini)}
    else
      Process.sleep(Keyword.get(opts, :poll_interval_ms, @poll_interval_ms))
      poll_once(ref, api_key, name, attempt, opts)
    end
  end

  @spec poll_once(FileRef.t(), String.t(), String.t(), non_neg_integer(), keyword()) ::
          {:ok, FileRef.t()} | {:error, Error.t()}
  defp poll_once(ref, api_key, name, attempt, opts) do
    base_url =
      Keyword.get(opts, :poll_base_url, "https://generativelanguage.googleapis.com/v1beta")

    req = Keyword.get(opts, :req, Req.new())

    Req.get(req, url: "#{base_url}/#{name}", headers: [{"x-goog-api-key", api_key}])
    |> Error.from_result(Gemini)
    |> case do
      {:ok, %{"state" => "ACTIVE"} = file_info} ->
        {:ok, build_file_ref(file_info, ref.mime_type, ref.filename)}

      {:ok, %{"state" => "PROCESSING"}} ->
        poll_until_active(ref, api_key, name, attempt + 1, opts)

      {:ok, %{"state" => "FAILED"}} ->
        {:error, Error.new(:server, "file processing failed", Gemini)}

      {:ok, body} ->
        {:error, Error.unexpected_response(body, Gemini)}

      {:error, _error} = failure ->
        failure
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
