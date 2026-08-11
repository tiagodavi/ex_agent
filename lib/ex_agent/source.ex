defmodule ExAgent.Source do
  @moduledoc """
  Pure helpers for reasoning about attachment sources.

  Infers MIME types from a path, a URL, or raw bytes, and maps a MIME type to
  the coarse modality providers gate on.

  This module knows nothing about providers and makes no decisions about how an
  attachment is delivered (inline, uploaded, or referenced by URL) — that is the
  responsibility of each provider's service module.
  """

  import Bitwise, only: [band: 2]

  @type modality :: :text | :image | :audio | :video | :document
  @type source :: {:path, String.t()} | {:url, String.t()} | {:data, binary()}

  @unknown_mime "application/octet-stream"

  # Media extensions the :mime registry does not carry. Configuring :mime globally
  # would force a recompile on every host application, so the fallback is local.
  @extra_extensions %{
    "flac" => "audio/flac",
    "m4a" => "audio/mp4",
    "m4b" => "audio/mp4"
  }

  @doc """
  Infers a MIME type from an attachment source.

  Paths and URLs are resolved by file extension (query strings and fragments are
  ignored); raw binaries are resolved by magic bytes. Returns an error naming
  `:mime_type` when the type cannot be determined, since passing it explicitly is
  always the fix.

  ## Examples

      iex> ExAgent.Source.infer_mime({:url, "https://cdn.example.com/a.png?sig=x"})
      {:ok, "image/png"}

      iex> ExAgent.Source.infer_mime({:data, <<0x89, "PNG", 0x0D, 0x0A, 0x1A, 0x0A>>})
      {:ok, "image/png"}

      iex> ExAgent.Source.infer_mime({:path, "/tmp/mystery"})
      {:error, "could not infer :mime_type from path \\"/tmp/mystery\\"; pass :mime_type explicitly"}
  """
  @spec infer_mime(source()) :: {:ok, String.t()} | {:error, String.t()}
  def infer_mime({:path, path}) when is_binary(path) do
    case from_extension(path) do
      {:ok, mime_type} -> {:ok, mime_type}
      :error -> {:error, cannot_infer("path", path)}
    end
  end

  def infer_mime({:url, url}) when is_binary(url) do
    # Only the path segment carries the extension; ?sig=... would otherwise
    # defeat extension lookup entirely.
    path = URI.parse(url).path || ""

    case from_extension(path) do
      {:ok, mime_type} -> {:ok, mime_type}
      :error -> {:error, cannot_infer("url", url)}
    end
  end

  def infer_mime({:data, data}) when is_binary(data) do
    case from_magic_bytes(data) do
      {:ok, mime_type} -> {:ok, mime_type}
      :error -> {:error, "could not infer :mime_type from data; pass :mime_type explicitly"}
    end
  end

  @doc """
  Maps a MIME type to the modality providers declare support for.

  Anything that is not image, audio, or video is treated as a document.

  ## Examples

      iex> ExAgent.Source.modality("video/mp4")
      :video

      iex> ExAgent.Source.modality("text/csv")
      :document
  """
  @spec modality(String.t()) :: modality()
  def modality("image/" <> _), do: :image
  def modality("audio/" <> _), do: :audio
  def modality("video/" <> _), do: :video
  def modality(_mime_type), do: :document

  @spec from_extension(String.t()) :: {:ok, String.t()} | :error
  defp from_extension(path) do
    case MIME.from_path(path) do
      @unknown_mime -> extra_extension(path)
      mime_type -> {:ok, mime_type}
    end
  end

  @spec extra_extension(String.t()) :: {:ok, String.t()} | :error
  defp extra_extension(path) do
    extension = path |> Path.extname() |> String.trim_leading(".") |> String.downcase()

    case Map.fetch(@extra_extensions, extension) do
      {:ok, mime_type} -> {:ok, mime_type}
      :error -> :error
    end
  end

  @spec from_magic_bytes(binary()) :: {:ok, String.t()} | :error
  defp from_magic_bytes(<<0x89, "PNG", 0x0D, 0x0A, 0x1A, 0x0A, _rest::binary>>),
    do: {:ok, "image/png"}

  defp from_magic_bytes(<<0xFF, 0xD8, 0xFF, _rest::binary>>), do: {:ok, "image/jpeg"}
  defp from_magic_bytes(<<"GIF8", _rest::binary>>), do: {:ok, "image/gif"}
  defp from_magic_bytes(<<"RIFF", _size::32, "WEBP", _rest::binary>>), do: {:ok, "image/webp"}
  defp from_magic_bytes(<<"RIFF", _size::32, "WAVE", _rest::binary>>), do: {:ok, "audio/wav"}
  defp from_magic_bytes(<<"ID3", _rest::binary>>), do: {:ok, "audio/mpeg"}
  defp from_magic_bytes(<<"fLaC", _rest::binary>>), do: {:ok, "audio/flac"}
  defp from_magic_bytes(<<"%PDF-", _rest::binary>>), do: {:ok, "application/pdf"}

  # ISO base media files share the `ftyp` box; only the brand says whether the
  # payload is video, audio, or QuickTime.
  defp from_magic_bytes(<<_size::32, "ftyp", brand::binary-size(4), _rest::binary>>),
    do: {:ok, ftyp_mime(brand)}

  # MPEG audio frame sync: 11 set bits.
  defp from_magic_bytes(<<0xFF, second, _rest::binary>>) when band(second, 0xE0) == 0xE0,
    do: {:ok, "audio/mpeg"}

  defp from_magic_bytes(_data), do: :error

  @spec ftyp_mime(binary()) :: String.t()
  defp ftyp_mime("qt  "), do: "video/quicktime"
  defp ftyp_mime("M4A "), do: "audio/mp4"
  defp ftyp_mime("M4B "), do: "audio/mp4"
  defp ftyp_mime(_brand), do: "video/mp4"

  @spec cannot_infer(String.t(), String.t()) :: String.t()
  defp cannot_infer(kind, value),
    do: "could not infer :mime_type from #{kind} #{inspect(value)}; pass :mime_type explicitly"
end
