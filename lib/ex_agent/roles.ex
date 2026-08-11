defmodule ExAgent.Roles do
  @moduledoc """
  Resolves provider structs from role names declared in application config.

  A *role* names a purpose (`:chat`, `:vision`, `:embed`, `:ocr`, ...) and maps it
  to a provider module plus the options its `new/1` takes. Application code then
  names the purpose instead of the vendor, so swapping a hosted model for a
  self-hosted container is a config change rather than an edit at every call site.

      # config/runtime.exs
      config :ex_agent, :roles,
        chat: {ExAgent.Providers.Gemini, api_key: System.fetch_env!("GEMINI_API_KEY")},
        vision: {ExAgent.Providers.OpenAICompatible,
                   base_url: System.fetch_env!("MODAL_QWEN_URL"),
                   model: "Qwen/Qwen3-VL-8B-Instruct",
                   modalities: [:text, :image]}

      ExAgent.provider!(:vision)   #=> %ExAgent.Providers.OpenAICompatible{}

  Role names are arbitrary atoms — the library has no fixed set. A bare module
  means `{module, []}`.

  ## Option values resolved at boot

  An option value that is a zero-arity function or an `{module, function, args}`
  tuple is invoked once during `build!/0`, for credentials that come from a vault
  rather than an env var:

      chat: {ExAgent.Providers.Gemini, api_key: {MyApp.Vault, :fetch, ["gemini"]}}

  Because of this, a literal three-element tuple cannot be used as an option
  value. No provider option currently takes one.

  ## Caching

  Roles are resolved **once**, during application start, and stored in
  `:persistent_term`. Reads are free — no copy, no table lookup — which matters
  because `ExAgent.provider!/1` sits on every request. Writes are not: each one
  triggers a global GC scan across every process in the VM, which is why
  `build!/0` is the only thing that writes and why per-call overrides
  (`ExAgent.provider!/2`) build a fresh struct instead of caching it.

  Call `build!/0` yourself if you assemble role config after boot.

  ## Security

  A cached provider struct holds a `Req` client with its API key baked into the
  request headers, so credentials live in `:persistent_term` and will appear in a
  VM crash dump. This is the same exposure as holding them in supervisor state,
  but it is worth knowing.
  """

  @roles_key {__MODULE__, :__roles__}
  @none :__ex_agent_no_role__

  @doc """
  Resolves every configured role into a provider struct and caches it.

  Raises `ArgumentError`, naming the offending role, if a spec is malformed, its
  module is missing, does not export `new/1`, does not implement
  `ExAgent.Provider`, or its `new/1` raises. Failing here means a missing
  credential crashes at deploy time rather than returning a 401 on the first
  request.

  Every role is built before anything is written, so a bad spec cannot leave a
  half-applied set behind. Rebuilding replaces the previous set entirely.
  """
  @spec build!() :: :ok
  def build! do
    specs = Application.get_env(:ex_agent, :roles, [])
    built = Enum.map(specs, fn {role, spec} -> {role, normalize!(role, spec)} end)

    Enum.each(list(), fn role ->
      :persistent_term.erase(key(role))
      :persistent_term.erase(spec_key(role))
    end)

    Enum.each(built, fn {role, {module, opts, provider}} ->
      :persistent_term.put(key(role), provider)
      :persistent_term.put(spec_key(role), {module, opts})
    end)

    :persistent_term.put(@roles_key, Enum.map(built, &elem(&1, 0)))
    :ok
  end

  @doc """
  Returns the cached provider struct for `role`, or `:error` if it is not configured.
  """
  @spec fetch(atom()) :: {:ok, struct()} | :error
  def fetch(role) when is_atom(role) do
    case :persistent_term.get(key(role), @none) do
      @none -> :error
      provider -> {:ok, provider}
    end
  end

  @doc """
  Returns the cached provider struct for `role`, raising if it is not configured.
  """
  @spec fetch!(atom()) :: struct()
  def fetch!(role) when is_atom(role) do
    case fetch(role) do
      {:ok, provider} -> provider
      :error -> raise ArgumentError, unknown_role_message(role)
    end
  end

  @doc """
  Builds a fresh provider struct for `role` with `overrides` merged over its
  configured options.

  The result is **not** cached — see the caching note in the moduledoc. Building
  reconstructs the `Req` client, which is fine per request but not inside a tight
  loop. Option values already resolved at boot are reused, so a vault-backed
  credential is not fetched again.
  """
  @spec build_with!(atom(), keyword()) :: struct()
  def build_with!(role, overrides) when is_atom(role) and is_list(overrides) do
    case :persistent_term.get(spec_key(role), @none) do
      @none -> raise ArgumentError, unknown_role_message(role)
      {module, opts} -> construct!(role, module, Keyword.merge(opts, overrides))
    end
  end

  @doc """
  Returns the configured role names, in declaration order.
  """
  @spec list() :: [atom()]
  def list, do: :persistent_term.get(@roles_key, [])

  # --- Building ---

  @spec normalize!(atom(), term()) :: {module(), keyword(), struct()}
  defp normalize!(role, module) when is_atom(module) and not is_nil(module),
    do: normalize!(role, {module, []})

  defp normalize!(role, {module, opts}) when is_atom(module) and is_list(opts) do
    ensure_provider!(role, module)
    opts = Enum.map(opts, fn {key, value} -> {key, resolve(value)} end)
    {module, opts, construct!(role, module, opts)}
  end

  defp normalize!(role, spec) do
    raise ArgumentError,
          "role #{inspect(role)} has an invalid spec: #{inspect(spec)}. " <>
            "Expected `Module` or `{Module, opts}`."
  end

  @spec construct!(atom(), module(), keyword()) :: struct()
  defp construct!(role, module, opts) do
    module.new(opts)
  rescue
    error ->
      reraise ArgumentError,
              [
                message:
                  "role #{inspect(role)} — #{inspect(module)}.new/1 raised: " <>
                    Exception.message(error)
              ],
              __STACKTRACE__
  end

  @spec ensure_provider!(atom(), module()) :: :ok
  defp ensure_provider!(role, module) do
    cond do
      not Code.ensure_loaded?(module) ->
        raise ArgumentError, "role #{inspect(role)} — module #{inspect(module)} is not available"

      not function_exported?(module, :new, 1) ->
        raise ArgumentError, "role #{inspect(role)} — #{inspect(module)} does not export new/1"

      ExAgent.Provider not in behaviours(module) ->
        raise ArgumentError,
              "role #{inspect(role)} — #{inspect(module)} does not implement " <>
                "the ExAgent.Provider behaviour"

      true ->
        :ok
    end
  end

  # A module may declare several behaviours, so every `behaviour` attribute counts.
  @spec behaviours(module()) :: [module()]
  defp behaviours(module) do
    module.module_info(:attributes) |> Keyword.get_values(:behaviour) |> List.flatten()
  end

  @spec resolve(term()) :: term()
  defp resolve(fun) when is_function(fun, 0), do: fun.()

  defp resolve({module, function, args})
       when is_atom(module) and is_atom(function) and is_list(args),
       do: apply(module, function, args)

  defp resolve(value), do: value

  # --- Keys and messages ---

  @spec key(atom()) :: {module(), :provider, atom()}
  defp key(role), do: {__MODULE__, :provider, role}

  @spec spec_key(atom()) :: {module(), :spec, atom()}
  defp spec_key(role), do: {__MODULE__, :spec, role}

  @spec unknown_role_message(atom()) :: String.t()
  defp unknown_role_message(role) do
    """
    no provider configured for role #{inspect(role)}

    Configured roles: #{inspect(list())}

    Add it in config/runtime.exs:

        config :ex_agent, :roles,
          #{role}: {ExAgent.Providers.Gemini, api_key: ...}
    """
  end
end
