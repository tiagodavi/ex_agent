defmodule ExAgent.UploadCacheTest do
  # The cache is a single named ETS table shared by the whole VM, and `clear/0`
  # is global — so this file must not run alongside other cache users.
  use ExUnit.Case, async: false

  alias ExAgent.{FileRef, UploadCache}
  alias ExAgent.Providers.{Gemini, OpenAI}

  setup do
    UploadCache.clear()
    :ok
  end

  defp openai_ref(file_id \\ "file-abc", expires_at \\ nil) do
    %FileRef{
      provider: :openai,
      file_id: file_id,
      mime_type: "application/pdf",
      filename: "report.pdf",
      expires_at: expires_at
    }
  end

  defp gemini_ref(expires_at) do
    %FileRef{
      provider: :gemini,
      file_uri: "files/xyz",
      mime_type: "video/mp4",
      expires_at: expires_at
    }
  end

  describe "scope/3" do
    test "given the same provider, url and key, then the scope is stable" do
      assert UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1") ==
               UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")
    end

    test "given different api keys, then the scopes differ" do
      refute UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1") ==
               UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-2")
    end

    test "given different base urls, then the scopes differ" do
      refute UploadCache.scope(OpenAI, "https://a.modal.run", "k") ==
               UploadCache.scope(OpenAI, "https://b.modal.run", "k")
    end

    test "given different providers, then the scopes differ" do
      refute UploadCache.scope(OpenAI, "https://x", "k") ==
               UploadCache.scope(Gemini, "https://x", "k")
    end

    test "given an api key, then it is not recoverable from the scope" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-supersecret")

      refute scope =~ "sk-supersecret"
      # A digest, not a concatenation.
      assert byte_size(scope) == 32
    end
  end

  describe "fetch/2 and put/3" do
    test "given nothing cached, then fetch misses" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")

      assert :miss = UploadCache.fetch(scope, "some bytes")
    end

    test "given a stored ref, then the same bytes hit" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")
      ref = openai_ref()

      assert :ok = UploadCache.put(scope, "some bytes", ref)
      assert {:ok, ^ref} = UploadCache.fetch(scope, "some bytes")
    end

    test "given different bytes in the same scope, then it misses" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")
      UploadCache.put(scope, "some bytes", openai_ref())

      assert :miss = UploadCache.fetch(scope, "other bytes")
    end

    test "given the same bytes under a different scope, then it misses" do
      scope_a = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-account-a")
      scope_b = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-account-b")

      UploadCache.put(scope_a, "shared bytes", openai_ref("file-from-a"))

      # Account B must never receive account A's file id.
      assert :miss = UploadCache.fetch(scope_b, "shared bytes")
    end

    test "given a re-put for the same key, then the newest ref wins" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")

      UploadCache.put(scope, "bytes", openai_ref("file-old"))
      UploadCache.put(scope, "bytes", openai_ref("file-new"))

      assert {:ok, %FileRef{file_id: "file-new"}} = UploadCache.fetch(scope, "bytes")
    end
  end

  describe "expiry" do
    test "given an unexpired ref, then it hits" do
      scope = UploadCache.scope(Gemini, "https://generativelanguage.googleapis.com", "AIza")
      future = DateTime.add(DateTime.utc_now(), 3600, :second)

      UploadCache.put(scope, "video bytes", gemini_ref(future))

      assert {:ok, %FileRef{file_uri: "files/xyz"}} = UploadCache.fetch(scope, "video bytes")
    end

    test "given an expired ref, then it misses so the caller re-uploads" do
      scope = UploadCache.scope(Gemini, "https://generativelanguage.googleapis.com", "AIza")
      past = DateTime.add(DateTime.utc_now(), -3600, :second)

      UploadCache.put(scope, "video bytes", gemini_ref(past))

      assert :miss = UploadCache.fetch(scope, "video bytes")
    end

    test "given an expired ref, then the entry is evicted rather than re-checked" do
      scope = UploadCache.scope(Gemini, "https://generativelanguage.googleapis.com", "AIza")
      past = DateTime.add(DateTime.utc_now(), -3600, :second)

      UploadCache.put(scope, "video bytes", gemini_ref(past))
      assert :miss = UploadCache.fetch(scope, "video bytes")

      assert UploadCache.size() == 0
    end

    test "given a ref with no expiry, then it never expires" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")

      UploadCache.put(scope, "bytes", openai_ref("file-forever", nil))

      assert {:ok, %FileRef{file_id: "file-forever"}} = UploadCache.fetch(scope, "bytes")
    end
  end

  describe "clear/0 and size/0" do
    test "given cached entries, then clear removes them all" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")

      UploadCache.put(scope, "a", openai_ref("file-a"))
      UploadCache.put(scope, "b", openai_ref("file-b"))
      assert UploadCache.size() == 2

      assert :ok = UploadCache.clear()

      assert UploadCache.size() == 0
      assert :miss = UploadCache.fetch(scope, "a")
    end
  end

  describe "concurrency" do
    test "given concurrent readers and writers, then no process is serialized behind the owner" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")

      results =
        1..50
        |> Task.async_stream(fn n ->
          bytes = "payload-#{n}"
          UploadCache.put(scope, bytes, openai_ref("file-#{n}"))
          UploadCache.fetch(scope, bytes)
        end)
        |> Enum.map(fn {:ok, result} -> result end)

      assert length(results) == 50
      assert Enum.all?(results, &match?({:ok, %FileRef{}}, &1))
      assert UploadCache.size() == 50
    end

    test "given the owner crashes, then the table is recreated" do
      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")
      UploadCache.put(scope, "bytes", openai_ref())

      pid = Process.whereis(UploadCache)
      ref = Process.monitor(pid)
      Process.exit(pid, :kill)
      assert_receive {:DOWN, ^ref, :process, ^pid, :killed}

      # Wait on the table, not the pid: the name is registered before init/1
      # creates the table, so whereis/1 alone is not enough.
      wait_until(fn -> :ets.whereis(UploadCache) != :undefined end)

      assert :miss = UploadCache.fetch(scope, "bytes")
      assert :ok = UploadCache.put(scope, "bytes", openai_ref())
      assert {:ok, %FileRef{}} = UploadCache.fetch(scope, "bytes")
    end
  end

  describe "degradation" do
    test "given no table, then operations degrade to misses instead of raising" do
      # A cache must never be the reason a request fails.
      pid = Process.whereis(UploadCache)
      :ok = GenServer.stop(pid, :normal)

      scope = UploadCache.scope(OpenAI, "https://api.openai.com/v1", "sk-1")

      assert :miss = UploadCache.fetch(scope, "bytes")
      assert :ok = UploadCache.put(scope, "bytes", openai_ref())
      assert :ok = UploadCache.clear()
      assert UploadCache.size() == 0

      wait_until(fn -> :ets.whereis(UploadCache) != :undefined end)
    end
  end

  defp wait_until(fun, attempts \\ 50)
  defp wait_until(_fun, 0), do: flunk("condition never became true")

  defp wait_until(fun, attempts) do
    if fun.() do
      :ok
    else
      Process.sleep(10)
      wait_until(fun, attempts - 1)
    end
  end
end
