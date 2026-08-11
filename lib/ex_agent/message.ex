defmodule ExAgent.Message do
  @moduledoc """
  Normalized message struct used across all LLM providers.

  Represents a single message in a conversation, supporting
  system, user, assistant, and tool roles.
  """

  @type role :: :system | :user | :assistant | :tool

  @type attachment :: ExAgent.Attachment.t()

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

  @doc """
  Applies `fun` to every attachment on every message, halting on the first error.

  Provider services use this to resolve attachments - loading bytes, uploading
  oversized files - before building a request body, so their formatting
  functions stay free of IO.
  """
  @spec map_attachments([t()], (attachment() -> {:ok, attachment()} | {:error, error})) ::
          {:ok, [t()]} | {:error, error}
        when error: term()
  def map_attachments(messages, fun) do
    collect(messages, fn
      %__MODULE__{attachments: []} = message ->
        {:ok, message}

      %__MODULE__{attachments: attachments} = message ->
        with {:ok, mapped} <- collect(attachments, fun) do
          {:ok, %{message | attachments: mapped}}
        end
    end)
  end

  @spec collect([item], (item -> {:ok, result} | {:error, error})) ::
          {:ok, [result]} | {:error, error}
        when item: term(), result: term(), error: term()
  defp collect(items, fun) do
    items
    |> Enum.reduce_while({:ok, []}, fn item, {:ok, acc} ->
      case fun.(item) do
        {:ok, result} -> {:cont, {:ok, [result | acc]}}
        {:error, _} = err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, results} -> {:ok, Enum.reverse(results)}
      {:error, _} = err -> err
    end
  end

  @spec validate_attachments([map()]) :: {:ok, [attachment()]} | {:error, String.t()}
  defp validate_attachments([]), do: {:ok, []}

  defp validate_attachments(attachments) when is_list(attachments),
    do: collect(attachments, &ExAgent.Attachment.new/1)

  defp validate_attachments(_), do: {:error, "attachments must be a list"}
end
