defmodule ExAgent.Message do
  @moduledoc """
  Normalized message struct used across all LLM providers.

  Represents a single message in a conversation, supporting
  system, user, assistant, and tool roles.
  """

  @type role :: :system | :user | :assistant | :tool

  @type attachment :: %{data: binary(), mime_type: String.t()}

  @type t :: %__MODULE__{
          role: role(),
          content: String.t(),
          tool_call_id: String.t() | nil,
          tool_calls: [map()] | nil,
          metadata: map(),
          attachments: [attachment()]
        }

  @enforce_keys [:role, :content]
  defstruct [:role, :content, :tool_call_id, :tool_calls, metadata: %{}, attachments: []]

  @valid_roles ~w(system user assistant tool)a

  @doc """
  Creates a new message with validated attributes.

  ## Examples

      iex> {:ok, msg} = ExAgent.Message.new(role: :user, content: "Hello")
      iex> msg.role
      :user

      iex> ExAgent.Message.new(role: :invalid, content: "Hello")
      {:error, "invalid role: :invalid. Must be one of: [:system, :user, :assistant, :tool]"}

      iex> ExAgent.Message.new(role: :user)
      {:error, "content is required"}
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, String.t()}
  def new(attrs) when is_list(attrs) do
    with {:ok, role} <- validate_role(attrs[:role]),
         {:ok, content} <- validate_content(attrs[:content]),
         {:ok, attachments} <- validate_attachments(attrs[:attachments] || []) do
      {:ok,
       %__MODULE__{
         role: role,
         content: content,
         tool_call_id: attrs[:tool_call_id],
         tool_calls: attrs[:tool_calls],
         metadata: attrs[:metadata] || %{},
         attachments: attachments
       }}
    end
  end

  @spec validate_role(atom() | nil) :: {:ok, role()} | {:error, String.t()}
  defp validate_role(nil), do: {:error, "role is required"}

  defp validate_role(role) when role in @valid_roles, do: {:ok, role}

  defp validate_role(role),
    do: {:error, "invalid role: #{inspect(role)}. Must be one of: #{inspect(@valid_roles)}"}

  @spec validate_content(String.t() | nil) :: {:ok, String.t()} | {:error, String.t()}
  defp validate_content(nil), do: {:error, "content is required"}
  defp validate_content(content) when is_binary(content), do: {:ok, content}
  defp validate_content(_), do: {:error, "content must be a string"}

  @spec validate_attachments([map()]) :: {:ok, [attachment()]} | {:error, String.t()}
  defp validate_attachments([]), do: {:ok, []}

  defp validate_attachments(attachments) when is_list(attachments) do
    Enum.reduce_while(attachments, {:ok, []}, fn att, {:ok, acc} ->
      case resolve_attachment(att) do
        {:ok, resolved} -> {:cont, {:ok, acc ++ [resolved]}}
        {:error, _} = err -> {:halt, err}
      end
    end)
  end

  defp validate_attachments(_), do: {:error, "attachments must be a list"}

  @spec resolve_attachment(map()) :: {:ok, attachment()} | {:error, String.t()}
  defp resolve_attachment(%{data: data, mime_type: mime_type})
       when is_binary(data) and is_binary(mime_type) do
    {:ok, %{data: data, mime_type: mime_type}}
  end

  defp resolve_attachment(%{path: path, mime_type: mime_type})
       when is_binary(path) and is_binary(mime_type) do
    case File.read(path) do
      {:ok, data} -> {:ok, %{data: data, mime_type: mime_type}}
      {:error, reason} -> {:error, "failed to read file #{path}: #{inspect(reason)}"}
    end
  end

  defp resolve_attachment(_),
    do: {:error, "each attachment must have :mime_type and either :data or :path"}
end
