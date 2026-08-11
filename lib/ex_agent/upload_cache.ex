defmodule ExAgent.UploadCache do
  @moduledoc """
  Caches provider file uploads so the same bytes are not uploaded twice.

  Uploading is slow and, on some providers, billed. When the same asset is
  attached across several turns of a conversation - or across conversations -
  the second and later sends should reuse the first upload's
  `ExAgent.FileRef`.

  Entries are keyed by `{scope, sha256(bytes)}`, where the scope digests the
  provider module, base URL, and API key. Scoping by API key matters: two
  provider structs for different accounts must never share a `file_id`, because
  one account's file reference is invalid - or worse, readable - for the other.
  The API key itself is never stored, only its digest.

  ## Expiry

  `fetch/2` treats an entry whose `ExAgent.FileRef` has expired as a miss and
  evicts it, so the caller uploads afresh.

  ## Growth

  There is no eviction beyond expiry. Entries are small (a `FileRef` struct and
  two digests), but an application uploading a large number of *distinct* files
  will grow the table without bound, since OpenAI file references never expire.
  Call `clear/0` if that matters for your workload.

  The table is public and named so that concurrent agents read it directly
  rather than serializing behind this process; the GenServer exists only to own
  the table across crashes.
  """

  use GenServer

  alias ExAgent.FileRef

  @table __MODULE__

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Builds the cache scope for a provider instance.

  Two provider structs share a scope only when their module, base URL, and API
  key all match. The API key is digested, never retained.
  """
  @spec scope(module(), String.t(), String.t() | nil) :: binary()
  def scope(provider_module, base_url, api_key) do
    :crypto.hash(:sha256, [inspect(provider_module), 0, base_url || "", 0, api_key || ""])
  end

  @doc """
  Looks up a cached `ExAgent.FileRef` for these bytes.

  Returns `:miss` when nothing is cached or the cached reference has expired.
  """
  @spec fetch(binary(), binary()) :: {:ok, FileRef.t()} | :miss
  def fetch(scope, bytes) do
    key = key(scope, bytes)

    with {:ok, table} <- table(),
         [{^key, %FileRef{} = ref}] <- :ets.lookup(table, key) do
      if FileRef.expired?(ref) do
        :ets.delete(table, key)
        :miss
      else
        {:ok, ref}
      end
    else
      _ -> :miss
    end
  end

  @doc """
  Stores an `ExAgent.FileRef` for these bytes, replacing any existing entry.
  """
  @spec put(binary(), binary(), FileRef.t()) :: :ok
  def put(scope, bytes, %FileRef{} = ref) do
    with {:ok, table} <- table() do
      :ets.insert(table, {key(scope, bytes), ref})
    end

    :ok
  end

  @doc """
  Removes every cached upload.
  """
  @spec clear() :: :ok
  def clear do
    with {:ok, table} <- table() do
      :ets.delete_all_objects(table)
    end

    :ok
  end

  @doc """
  Returns the number of cached entries, including any not yet evicted expired ones.
  """
  @spec size() :: non_neg_integer()
  def size do
    case table() do
      {:ok, table} -> :ets.info(table, :size)
      :error -> 0
    end
  end

  # A cache must never be the reason a request fails. If the owner is restarting,
  # every operation degrades to a miss rather than raising into the caller.
  @spec table() :: {:ok, :ets.table()} | :error
  defp table do
    case :ets.whereis(@table) do
      :undefined -> :error
      table -> {:ok, table}
    end
  end

  @impl true
  def init(_opts) do
    :ets.new(@table, [
      :set,
      :named_table,
      :public,
      read_concurrency: true,
      write_concurrency: true
    ])

    {:ok, %{}}
  end

  @spec key(binary(), binary()) :: {binary(), binary()}
  defp key(scope, bytes), do: {scope, :crypto.hash(:sha256, bytes)}
end
