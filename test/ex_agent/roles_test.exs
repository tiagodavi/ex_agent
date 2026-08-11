defmodule ExAgent.RolesTest do
  @moduledoc """
  Roles let an application declare *which provider serves which purpose* in
  config and resolve it by atom at the call site. They must be additive sugar:
  `provider!/1` returns a struct usable everywhere a hand-built one is, and every
  existing entry point keeps working untouched.
  """

  # Roles live in `:persistent_term`, which is VM-global.
  use ExUnit.Case, async: false

  alias ExAgent.{Embeddings, Error, Response, Roles}

  # --- Test doubles ---

  defmodule Fake do
    @moduledoc false
    @behaviour ExAgent.Provider

    defstruct [:api_key, :model, :reply, tools: []]

    def new(opts) do
      %__MODULE__{
        api_key: opts[:api_key] || "default-key",
        model: opts[:model] || "fake-1",
        reply: opts[:reply],
        tools: opts[:tools] || []
      }
    end

    @impl true
    def chat(%__MODULE__{reply: {:tool_call, _, _} = reply}, _messages, _opts), do: reply

    def chat(%__MODULE__{model: model}, messages, _opts) do
      {:ok, message} =
        ExAgent.Message.new(role: :assistant, content: "#{model}/#{length(messages)}")

      {:ok, Response.new(message)}
    end

    @impl true
    def stream(%__MODULE__{model: model}, _messages, _opts),
      do: [ExAgent.Chunk.text_delta(model), ExAgent.Chunk.done()]
  end

  defmodule Embedder do
    @moduledoc false
    @behaviour ExAgent.Provider

    defstruct model: "embed-1"

    def new(opts), do: struct!(__MODULE__, opts)

    @impl true
    def chat(_provider, _messages, _opts), do: {:error, Error.new(:unsupported, "chat only")}

    @impl true
    def embed(%__MODULE__{model: model}, inputs, _opts) do
      {:ok,
       %Embeddings{
         vectors: Enum.map(inputs, fn _ -> [1.0, 0.0] end),
         model: model,
         provider: __MODULE__,
         dimensions: 2
       }}
    end
  end

  defmodule NotAProvider do
    @moduledoc false
    def new(_opts), do: %{}
  end

  defmodule WithoutNew do
    @moduledoc false
    @behaviour ExAgent.Provider

    @impl true
    def chat(_provider, _messages, _opts), do: {:error, Error.new(:server, "never called")}
  end

  defmodule Vault do
    @moduledoc false
    def fetch(name), do: "secret-for-#{name}"
  end

  # --- Helpers ---

  defp configure!(roles) do
    Application.put_env(:ex_agent, :roles, roles)
    Roles.build!()
  end

  setup do
    on_exit(fn ->
      Application.delete_env(:ex_agent, :roles)
      Roles.build!()
    end)

    :ok
  end

  describe "provider!/1" do
    test "given a configured role, then the struct is built through the module's new/1" do
      configure!(chat: {Fake, model: "gemini-ish", api_key: "k"})

      assert %Fake{model: "gemini-ish", api_key: "k"} = ExAgent.provider!(:chat)
    end

    test "given a bare module, then it is treated as an empty option list" do
      configure!(chat: Fake)

      assert %Fake{model: "fake-1", api_key: "default-key"} = ExAgent.provider!(:chat)
    end

    test "given repeated calls, then the same cached struct comes back" do
      configure!(chat: {Fake, model: "cached"})

      assert ExAgent.provider!(:chat) === ExAgent.provider!(:chat)
    end

    test "given an unknown role, then the error names the configured ones" do
      configure!(chat: Fake, vision: Fake)

      message = assert_raise(ArgumentError, fn -> ExAgent.provider!(:visoin) end).message

      assert message =~ "no provider configured for role :visoin"
      assert message =~ "Configured roles: [:chat, :vision]"
      assert message =~ "config :ex_agent, :roles"
    end
  end

  describe "provider/1 and roles/0" do
    test "given a configured role, then provider/1 wraps the struct in :ok" do
      configure!(chat: Fake)

      assert {:ok, %Fake{}} = ExAgent.provider(:chat)
    end

    test "given an unknown role, then provider/1 returns :error instead of raising" do
      configure!(chat: Fake)

      assert :error = ExAgent.provider(:nope)
    end

    test "given several roles, then roles/0 lists them in declaration order" do
      configure!(chat: Fake, vision: Fake, embed: Embedder)

      assert ExAgent.roles() == [:chat, :vision, :embed]
    end

    test "given no role config at all, then build!/0 is a no-op and roles/0 is empty" do
      Application.delete_env(:ex_agent, :roles)
      assert Roles.build!() == :ok

      assert ExAgent.roles() == []
      assert ExAgent.provider(:chat) == :error
    end
  end

  describe "option values resolved at boot" do
    test "given an MFA option value, then it is applied once during build!/0" do
      configure!(chat: {Fake, api_key: {Vault, :fetch, ["gemini"]}})

      assert %Fake{api_key: "secret-for-gemini"} = ExAgent.provider!(:chat)
    end

    test "given a zero-arity function option value, then it is invoked during build!/0" do
      configure!(chat: {Fake, api_key: fn -> "from-fun" end})

      assert %Fake{api_key: "from-fun"} = ExAgent.provider!(:chat)
    end

    test "given an override, then the boot-resolved value is reused, not re-fetched" do
      # A vault-backed credential must not be hit again on every overridden call.
      test_pid = self()

      configure!(chat: {Fake, api_key: fn -> send(test_pid, :fetched) && "v1" end})
      assert_received :fetched

      assert %Fake{api_key: "v1", model: "other"} = ExAgent.provider!(:chat, model: "other")
      refute_received :fetched
    end
  end

  describe "provider!/2 overrides" do
    setup do
      configure!(chat: {Fake, model: "base", api_key: "base-key"})
    end

    test "given overrides, then they win over the configured options" do
      assert %Fake{model: "tenant", api_key: "base-key"} =
               ExAgent.provider!(:chat, model: "tenant")
    end

    test "given overrides, then the cached struct is left untouched" do
      _overridden = ExAgent.provider!(:chat, model: "tenant")

      assert %Fake{model: "base"} = ExAgent.provider!(:chat)
    end

    test "given an empty override list, then the cached struct is returned as-is" do
      assert ExAgent.provider!(:chat, []) === ExAgent.provider!(:chat)
    end

    test "given an unknown role, then overrides raise the same message as provider!/1" do
      assert_raise ArgumentError, ~r/no provider configured for role :nope/, fn ->
        ExAgent.provider!(:nope, model: "x")
      end
    end
  end

  describe "caching" do
    test "given repeated lookups, then no :persistent_term writes occur" do
      # Every write triggers a global GC scan, so a per-call write would be a
      # latency bug that only appears under load.
      configure!(chat: Fake)

      before = :persistent_term.info()[:count]
      Enum.each(1..100, fn _ -> ExAgent.provider!(:chat) end)

      assert :persistent_term.info()[:count] == before
    end

    test "given a rebuild, then roles removed from config are dropped" do
      configure!(chat: Fake, vision: Fake)
      configure!(chat: Fake)

      assert ExAgent.roles() == [:chat]
      assert ExAgent.provider(:vision) == :error
    end
  end

  describe "boot validation" do
    test "given a module that is not available, then build!/0 names the role" do
      assert_raise ArgumentError, ~r/role :chat - module .*Missing is not available/, fn ->
        configure!(chat: ExAgent.RolesTest.Missing)
      end
    end

    test "given a module without new/1, then build!/0 names the role" do
      assert_raise ArgumentError, ~r/role :chat - .*WithoutNew does not export new\/1/, fn ->
        configure!(chat: WithoutNew)
      end
    end

    test "given a module that does not implement the behaviour, then build!/0 rejects it" do
      assert_raise ArgumentError, ~r/does not implement the ExAgent.Provider behaviour/, fn ->
        configure!(chat: NotAProvider)
      end
    end

    test "given new/1 raising, then the role and the underlying reason are both reported" do
      # A missing credential must crash at deploy time, not 401 at 3am.
      message =
        assert_raise(ArgumentError, fn ->
          configure!(chat: ExAgent.Providers.OpenAI)
        end).message

      assert message =~ "role :chat - ExAgent.Providers.OpenAI.new/1 raised:"
      assert message =~ "api_key"
    end

    test "given a malformed spec, then build!/0 explains the expected shape" do
      assert_raise ArgumentError, ~r/invalid spec: "nope".*Expected `Module`/s, fn ->
        configure!(chat: "nope")
      end
    end

    test "given one bad role among good ones, then nothing is written" do
      configure!(chat: Fake)

      assert_raise ArgumentError, fn -> configure!(vision: Fake, broken: NotAProvider) end

      # The previous set survives intact rather than being half-replaced.
      assert ExAgent.roles() == [:chat]
      assert ExAgent.provider(:vision) == :error
    end
  end

  describe "chat_with/3" do
    test "given a role, then the provider is called with a single user message" do
      configure!(chat: {Fake, model: "m"})

      assert {:ok, %Response{content: "m/1"}} = ExAgent.chat_with(:chat, "hello")
    end

    test "given a tool-configured provider, then the raw tool call comes back unexecuted" do
      # Documented contract: chat_with/3 is one-shot; the tool loop lives in the agent.
      configure!(chat: {Fake, reply: {:tool_call, "search", %{"q" => "elixir"}}})

      assert {:tool_call, "search", %{"q" => "elixir"}} = ExAgent.chat_with(:chat, "hi")
    end

    test "given an invalid message, then an error tuple comes back before any dispatch" do
      configure!(chat: Fake)

      assert {:error, %Error{type: :invalid_request}} =
               ExAgent.chat_with(:chat, "hi", files: [%{}])
    end

    test "given an unknown role, then it raises the resolution error" do
      configure!(chat: Fake)

      assert_raise ArgumentError, ~r/no provider configured for role :vision/, fn ->
        ExAgent.chat_with(:vision, "hi")
      end
    end
  end

  describe "stream_with/3" do
    test "given a role, then chunks come from the role's provider" do
      configure!(chat: {Fake, model: "streamer"})

      assert {:ok, %Response{content: "streamer"}} =
               :chat |> ExAgent.stream_with("hi") |> ExAgent.collect()
    end

    test "given an invalid message, then a single terminal error chunk is yielded" do
      configure!(chat: Fake)

      assert [%ExAgent.Chunk{type: :done, finish_reason: :error}] =
               ExAgent.stream_with(:chat, "hi", files: [%{}])
    end
  end

  describe "embed_with/3" do
    test "given an embedding role, then vectors come back" do
      configure!(embed: {Embedder, model: "jina"})

      assert {:ok, %Embeddings{vectors: [_, _], model: "jina"}} =
               ExAgent.embed_with(:embed, ["a", "b"])
    end

    test "given a provider without the callback, then :unsupported comes back" do
      # This is why no role name is special-cased at boot: the dispatcher already
      # fails loudly for a provider with no embeddings endpoint.
      configure!(embed: Fake)

      assert {:error, %Error{type: :unsupported}} = ExAgent.embed_with(:embed, ["a"])
    end
  end

  describe "start_agent/1 with :role" do
    test "given a role, then the agent runs against that provider" do
      configure!(chat: {Fake, model: "agentic"})

      {:ok, agent} = ExAgent.start_agent(role: :chat)
      on_exit(fn -> ExAgent.stop_agent(agent) end)

      assert {:ok, %Response{content: "agentic/1"}} = ExAgent.chat(agent, "hi")
    end

    test "given both :role and :provider, then it raises" do
      configure!(chat: Fake)

      assert_raise ArgumentError, ~r/:provider or :role, not both/, fn ->
        ExAgent.Agent.start_link(role: :chat, provider: Fake.new([]))
      end
    end

    test "given an unknown role, then it raises the resolution error" do
      configure!(chat: Fake)

      assert_raise ArgumentError, ~r/no provider configured for role :nope/, fn ->
        ExAgent.Agent.start_link(role: :nope)
      end
    end

    test "given :provider, then the existing path is untouched" do
      configure!(chat: Fake)

      {:ok, agent} = ExAgent.start_agent(provider: Fake.new(model: "explicit"))
      on_exit(fn -> ExAgent.stop_agent(agent) end)

      assert {:ok, %Response{content: "explicit/1"}} = ExAgent.chat(agent, "hi")
    end
  end
end
