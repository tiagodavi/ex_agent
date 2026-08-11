defmodule ExAgent.Attachment do
  @moduledoc """
  A normalized file attachment carried on `ExAgent.Message`.

  Users never build this directly — pass plain maps under the `:files` option and
  `ExAgent.Message.new/1` normalizes them:

      files: [
        %{url: "https://cdn.example.com/invoice.png"},
        %{path: "/tmp/report.pdf"},
        %{data: png_bytes, mime_type: "image/png"},
        %{file_ref: ref}
      ]

  `:mime_type` is optional and inferred from the file extension (`:path`, `:url`)
  or magic bytes (`:data`). Supply it to override inference — an explicit value is
  always authoritative.

  Because a struct is also a map, provider services can keep pattern matching on
  `%{data: data, mime_type: mime_type}` and `%{file_ref: %ExAgent.FileRef{}}`.

  ## Provider options

  `:fps` and `:max_frames` (video sampling) are lifted into `:provider_opts`,
  alongside anything passed under an explicit `:provider_opts` map. They are
  normalized here but only meaningful to providers that support video.
  """

  alias ExAgent.{FileRef, Source}

  @type kind :: :data | :path | :url | :file_ref

  @type t :: %__MODULE__{
          kind: kind(),
          data: binary() | nil,
          path: String.t() | nil,
          url: String.t() | nil,
          file_ref: FileRef.t() | nil,
          mime_type: String.t(),
          filename: String.t() | nil,
          byte_size: non_neg_integer() | nil,
          modality: Source.modality(),
          provider_opts: map()
        }

  @enforce_keys [:kind, :mime_type, :modality]
  defstruct [
    :kind,
    :data,
    :path,
    :url,
    :file_ref,
    :mime_type,
    :filename,
    :byte_size,
    :modality,
    provider_opts: %{}
  ]

  @provider_opt_keys [:fps, :max_frames]

  @doc """
  Normalizes a user-supplied attachment map into an `%ExAgent.Attachment{}`.

  Exactly one source key (`:data`, `:path`, `:url`, `:file_ref`) must be present.

  ## Examples

      iex> {:ok, att} = ExAgent.Attachment.new(%{url: "https://cdn.example.com/a.png"})
      iex> {att.kind, att.mime_type, att.modality}
      {:url, "image/png", :image}

      iex> ExAgent.Attachment.new(%{})
      {:error, "each attachment must have one of :data, :path, :url, or :file_ref"}
  """
  @spec new(map() | t()) :: {:ok, t()} | {:error, String.t()}
  def new(%__MODULE__{} = attachment), do: {:ok, attachment}

  def new(%{file_ref: %FileRef{} = ref} = att) do
    build(:file_ref, ref.mime_type, att, file_ref: ref, filename: ref.filename)
  end

  def new(%{data: data} = att) when is_binary(data) do
    with {:ok, mime_type} <- resolve_mime(att, {:data, data}) do
      build(:data, mime_type, att, data: data, byte_size: byte_size(data))
    end
  end

  # Only the size is read up front — it is what the provider needs to choose
  # between inlining and uploading. Reading the bytes is deferred to request
  # time so a large file is not held in conversation history for every turn.
  def new(%{path: path} = att) when is_binary(path) do
    with {:ok, mime_type} <- resolve_mime(att, {:path, path}),
         {:ok, size} <- file_size(path) do
      build(:path, mime_type, att,
        path: path,
        byte_size: size,
        filename: Path.basename(path)
      )
    end
  end

  def new(%{url: url} = att) when is_binary(url) do
    with {:ok, mime_type} <- resolve_mime(att, {:url, url}) do
      build(:url, mime_type, att, url: url, filename: url_filename(url))
    end
  end

  def new(_att),
    do: {:error, "each attachment must have one of :data, :path, :url, or :file_ref"}

  @doc """
  Returns the attachment's bytes, reading from disk if they are not yet loaded.

  Returns an error for `:url` and `:file_ref` attachments, which have no local
  bytes by design.
  """
  @spec bytes(t()) :: {:ok, binary()} | {:error, ExAgent.Error.t()}
  def bytes(%__MODULE__{data: data}) when is_binary(data), do: {:ok, data}

  def bytes(%__MODULE__{kind: :path, path: path}) do
    case File.read(path) do
      {:ok, data} -> {:ok, data}
      {:error, reason} -> {:error, read_error(path, reason)}
    end
  end

  def bytes(%__MODULE__{kind: kind}) do
    {:error,
     ExAgent.Error.new(:invalid_request, "#{inspect(kind)} attachments have no local bytes")}
  end

  @doc """
  Loads a `:path` attachment's bytes into `:data`.

  `:url` and `:file_ref` attachments pass through untouched — they are delivered
  by reference and must never be fetched.
  """
  @spec load(t()) :: {:ok, t()} | {:error, ExAgent.Error.t()}
  def load(%__MODULE__{data: data} = attachment) when is_binary(data), do: {:ok, attachment}

  def load(%__MODULE__{kind: :path} = attachment) do
    with {:ok, data} <- bytes(attachment), do: {:ok, %{attachment | data: data}}
  end

  def load(%__MODULE__{} = attachment), do: {:ok, attachment}

  @spec build(kind(), String.t(), map(), keyword()) :: {:ok, t()}
  defp build(kind, mime_type, att, fields) do
    attachment =
      struct!(
        __MODULE__,
        [
          kind: kind,
          mime_type: mime_type,
          modality: Source.modality(mime_type),
          provider_opts: provider_opts(att)
        ] ++ fields
      )

    {:ok, override_filename(attachment, att)}
  end

  # An explicit :mime_type always wins; inference is the fallback.
  @spec resolve_mime(map(), Source.source()) :: {:ok, String.t()} | {:error, String.t()}
  defp resolve_mime(%{mime_type: mime_type}, _source) when is_binary(mime_type),
    do: {:ok, mime_type}

  defp resolve_mime(_att, source), do: Source.infer_mime(source)

  @spec file_size(String.t()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  defp file_size(path) do
    case File.stat(path) do
      {:ok, %File.Stat{size: size}} -> {:ok, size}
      {:error, reason} -> {:error, "failed to read file #{path}: #{inspect(reason)}"}
    end
  end

  @spec read_error(String.t(), term()) :: ExAgent.Error.t()
  defp read_error(path, reason),
    do: ExAgent.Error.new(:invalid_request, "failed to read file #{path}: #{inspect(reason)}")

  @spec override_filename(t(), map()) :: t()
  defp override_filename(attachment, %{filename: filename}) when is_binary(filename),
    do: %{attachment | filename: filename}

  defp override_filename(attachment, _att), do: attachment

  @spec provider_opts(map()) :: map()
  defp provider_opts(att) do
    att
    |> Map.get(:provider_opts, %{})
    |> Map.merge(Map.take(att, @provider_opt_keys))
  end

  @spec url_filename(String.t()) :: String.t() | nil
  defp url_filename(url) do
    case URI.parse(url).path do
      nil -> nil
      "" -> nil
      path -> Path.basename(path)
    end
  end
end
