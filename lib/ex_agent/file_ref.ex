defmodule ExAgent.FileRef do
  @moduledoc """
  Reference to a file previously uploaded to an LLM provider.

  Holds the provider-specific file identifier (OpenAI `file_id`, Gemini
  `file_uri`, or whichever a custom provider uses) so that uploaded files can be
  referenced in
  chat messages without re-sending the binary data.

  ## Examples

      iex> {:ok, ref} = ExAgent.FileRef.new(provider: :openai, file_id: "file-abc123", mime_type: "application/pdf")
      iex> ref.file_id
      "file-abc123"

      iex> {:ok, ref} = ExAgent.FileRef.new(provider: :gemini, file_uri: "https://example.com/files/abc", mime_type: "image/png")
      iex> ref.file_uri
      "https://example.com/files/abc"

      iex> ExAgent.FileRef.new(provider: :openai, mime_type: "image/png")
      {:error, "OpenAI file references require :file_id"}

      iex> ExAgent.FileRef.new(provider: :gemini, mime_type: "image/png")
      {:error, "Gemini file references require :file_uri"}
  """

  @type t :: %__MODULE__{
          provider: atom(),
          file_id: String.t() | nil,
          file_uri: String.t() | nil,
          mime_type: String.t(),
          filename: String.t() | nil,
          expires_at: DateTime.t() | nil
        }

  @enforce_keys [:provider, :mime_type]
  defstruct [:provider, :file_id, :file_uri, :mime_type, :filename, :expires_at]

  @doc """
  Creates a new file reference with validated attributes.

  ## Options

  - `:provider` (required) - `:openai`, `:gemini`, or the atom naming a custom
    provider that implements `c:ExAgent.Provider.upload/4`
  - `:mime_type` (required) - MIME type of the uploaded file
  - `:file_id` - OpenAI file ID (required for `:openai` provider)
  - `:file_uri` - Gemini file URI (required for `:gemini` provider)
  - `:filename` - Original filename
  - `:expires_at` - Expiration datetime (Gemini files expire after 48h)
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, String.t()}
  def new(attrs) when is_list(attrs) do
    with {:ok, provider} <- validate_provider(attrs[:provider]),
         {:ok, mime_type} <- validate_mime_type(attrs[:mime_type]),
         :ok <- validate_provider_fields(provider, attrs) do
      {:ok,
       %__MODULE__{
         provider: provider,
         file_id: attrs[:file_id],
         file_uri: attrs[:file_uri],
         mime_type: mime_type,
         filename: attrs[:filename],
         expires_at: attrs[:expires_at]
       }}
    end
  end

  @doc """
  Returns `true` if the file reference has expired.

  Gemini files expire 48 hours after upload. OpenAI files do not expire.

  ## Examples

      iex> {:ok, ref} = ExAgent.FileRef.new(provider: :openai, file_id: "f-1", mime_type: "text/plain")
      iex> ExAgent.FileRef.expired?(ref)
      false
  """
  @spec expired?(t()) :: boolean()
  def expired?(%__MODULE__{expires_at: nil}), do: false

  def expired?(%__MODULE__{expires_at: %DateTime{} = dt}) do
    DateTime.compare(DateTime.utc_now(), dt) == :gt
  end

  @spec validate_provider(atom() | nil) :: {:ok, atom()} | {:error, String.t()}
  defp validate_provider(nil), do: {:error, "provider is required"}

  # Any atom: a custom provider implementing `c:ExAgent.Provider.upload/4` has to
  # be able to build the reference its own callback returns. The field rules
  # below still apply to the built-ins.
  defp validate_provider(provider) when is_atom(provider), do: {:ok, provider}

  defp validate_provider(provider),
    do: {:error, "provider must be an atom, got: #{inspect(provider)}"}

  @spec validate_mime_type(String.t() | nil) :: {:ok, String.t()} | {:error, String.t()}
  defp validate_mime_type(nil), do: {:error, "mime_type is required"}
  defp validate_mime_type(mt) when is_binary(mt), do: {:ok, mt}
  defp validate_mime_type(_), do: {:error, "mime_type must be a string"}

  @spec validate_provider_fields(atom(), keyword()) :: :ok | {:error, String.t()}
  defp validate_provider_fields(:openai, attrs) do
    if is_binary(attrs[:file_id]) and attrs[:file_id] != "",
      do: :ok,
      else: {:error, "OpenAI file references require :file_id"}
  end

  defp validate_provider_fields(:gemini, attrs) do
    if is_binary(attrs[:file_uri]) and attrs[:file_uri] != "",
      do: :ok,
      else: {:error, "Gemini file references require :file_uri"}
  end

  # Whichever identifier a custom provider's API uses, a reference has to carry
  # one — that is the only rule this struct can state without knowing the API.
  defp validate_provider_fields(_provider, attrs) do
    if present?(attrs[:file_id]) or present?(attrs[:file_uri]),
      do: :ok,
      else: {:error, "file references require :file_id or :file_uri"}
  end

  @spec present?(term()) :: boolean()
  defp present?(value), do: is_binary(value) and value != ""
end
