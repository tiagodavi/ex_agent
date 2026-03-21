defmodule ExAgent.Patterns.Subagents do
  @moduledoc """
  Centralized orchestration pattern.

  A main agent invokes specialized subagents as tool calls. Subagents
  are transient, stateless processes with isolated contexts — they run
  a single LLM call and return the result without maintaining state.
  """

  alias ExAgent.{LlmProvider, Message, Tool}

  @type subagent_spec :: %{
          name: String.t(),
          description: String.t(),
          provider: struct(),
          system_prompt: String.t() | nil,
          tools: [Tool.t()]
        }

  @doc """
  Converts subagent specifications into Tool structs for use by an orchestrator agent.

  Each subagent becomes a tool whose function spawns an ephemeral LLM call.
  The tool's parameters accept a `"query"` string.
  """
  @spec build_orchestrator_tools([subagent_spec()]) :: [Tool.t()]
  def build_orchestrator_tools(specs) do
    Enum.map(specs, fn spec ->
      %Tool{
        name: spec.name,
        description: spec.description,
        parameters: %{
          "type" => "object",
          "properties" => %{
            "query" => %{"type" => "string", "description" => "The input for the subagent"}
          },
          "required" => ["query"]
        },
        function: fn %{"query" => query} -> invoke_subagent(spec, query) end
      }
    end)
  end

  @doc """
  Invokes a subagent synchronously with a single query.

  Creates a fresh context, calls `LlmProvider.chat/3` directly
  (no GenServer), and returns the assistant's response content.
  """
  @spec invoke_subagent(subagent_spec(), String.t()) :: {:ok, String.t()} | {:error, term()}
  def invoke_subagent(spec, query) do
    {:ok, user_msg} = Message.new(role: :user, content: query)
    provider = maybe_set_system_prompt(spec.provider, spec[:system_prompt])
    provider = %{provider | tools: spec[:tools] || []}

    case LlmProvider.chat(provider, [user_msg], []) do
      {:ok, %Message{content: content}} -> {:ok, content}
      {:error, reason} -> {:error, reason}
      {:tool_call, _name, _args} -> {:ok, "Subagent requested a tool call but tools are not executed in subagent mode."}
    end
  end

  @doc """
  Invokes multiple subagents in parallel using Task.async_stream.

  Returns a list of `{spec_name, result}` tuples.
  """
  @spec invoke_subagents_parallel([{subagent_spec(), String.t()}], keyword()) ::
          [{String.t(), {:ok, String.t()} | {:error, term()}}]
  def invoke_subagents_parallel(specs_with_inputs, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 30_000)

    specs_with_inputs
    |> Task.async_stream(
      fn {spec, input} -> {spec.name, invoke_subagent(spec, input)} end,
      timeout: timeout,
      on_timeout: :kill_task
    )
    |> Enum.map(fn
      {:ok, {name, result}} -> {name, result}
      {:exit, :timeout} -> {"unknown", {:error, :timeout}}
    end)
  end

  @spec maybe_set_system_prompt(struct(), String.t() | nil) :: struct()
  defp maybe_set_system_prompt(provider, nil), do: provider
  defp maybe_set_system_prompt(provider, prompt), do: %{provider | system_prompt: prompt}
end
