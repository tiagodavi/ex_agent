defprotocol ExAgent.FileUploader do
  @moduledoc """
  Protocol for uploading files to LLM providers.

  Providers that support file uploads (OpenAI, Gemini) implement this
  protocol to upload files and return a `FileRef` that can be used in
  subsequent chat messages.

  ## Implementations

  - `ExAgent.Providers.OpenAI` — uploads via `POST /v1/files`
  - `ExAgent.Providers.Gemini` — uploads via Gemini Files API

  ## Example

      provider = ExAgent.Providers.OpenAI.new(api_key: "sk-...")
      {:ok, ref} = ExAgent.FileUploader.upload(provider, file_data, "application/pdf", filename: "report.pdf")
  """

  @doc """
  Uploads binary file data to the provider and returns a file reference.

  ## Options

  - `:filename` - original filename (defaults to `"upload"`)
  - `:purpose` - OpenAI-specific file purpose (defaults to `"user_data"`)
  """
  @spec upload(t(), binary(), String.t(), keyword()) ::
          {:ok, ExAgent.FileRef.t()} | {:error, term()}
  def upload(provider, file_data, mime_type, opts \\ [])
end
