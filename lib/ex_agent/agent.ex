defmodule ExAgent.Agent do
  @moduledoc """
  GenServer that manages a single LLM agent.

  Holds the provider struct in state and dispatches calls via the
  `ExAgent.LlmProvider` protocol. Contains the tool execution loop
  that automatically invokes tools when the LLM requests them.

  ## State

  The agent state includes:
  - `provider` - any struct implementing `ExAgent.LlmProvider`
  - `context` - conversation history (`ExAgent.Context`)
  - `tools` - available tools for function-calling
  - `skills` - loadable skill definitions
  - `active_skill` - currently active skill (if any)
  """

  use GenServer

  alias ExAgent.{Context, LlmProvider, Message, Skill, Tool}
  alias ExAgent.Patterns.Skills, as: SkillsPattern

  @type agent_opts :: [
          id: String.t(),
          provider: struct(),
          tools: [Tool.t()],
          skills: [Skill.t()],
          name: GenServer.name()
        ]

  @type state :: %{
          id: String.t(),
          provider: struct(),
          context: Context.t(),
          tools: [Tool.t()],
          skills: [Skill.t()],
          active_skill: Skill.t() | nil,
          built_in_tools: [atom()],
          status: :idle | :processing | :handed_off
        }

  @max_tool_iterations 10

  # Client API

  @doc """
  Starts an agent process linked to the current process.
  """
  @spec start_link(agent_opts()) :: GenServer.on_start()
  def start_link(opts) do
    name = opts[:name]
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Sends a user message to the agent and returns the response.

  The agent appends the message to its context, evaluates skills,
  calls the LLM, and executes any requested tools in a loop until
  a final response or handoff is returned.

  ## Options

  - `:files` - list of file attachments, each a map with `:mime_type` and
    either `:data` (binary) or `:path` (file path). Files become part of the
    conversation context and are sent to the LLM alongside the text message.
  - `:built_in_tools` - list of provider-specific built-in tools to enable
    for this message. Overrides the agent-level `built_in_tools` if provided.
    Examples: `[:google_search]`, `[:web_search]`, `[:thinking]`.

  ## Examples

      ExAgent.Agent.chat(agent, "Describe this image",
        files: [%{path: "photo.jpg", mime_type: "image/jpeg"}])

      ExAgent.Agent.chat(agent, "Search the web for Elixir news",
        built_in_tools: [:google_search])
  """
  @spec chat(GenServer.server(), String.t(), keyword()) ::
          {:ok, Message.t()}
          | {:handoff, pid() | atom(), Context.t()}
          | {:error, term()}
  def chat(agent, user_input, opts \\ []) when is_binary(user_input) do
    GenServer.call(agent, {:chat, user_input, opts}, :infinity)
  end

  @doc """
  Sends a user message asynchronously, returning a Task.

  Accepts the same options as `chat/3`.
  """
  @spec chat_async(GenServer.server(), String.t(), keyword()) :: Task.t()
  def chat_async(agent, user_input, opts \\ []) do
    Task.Supervisor.async(ExAgent.TaskSupervisor, fn ->
      chat(agent, user_input, opts)
    end)
  end

  @doc """
  Returns the current conversation context.
  """
  @spec get_context(GenServer.server()) :: Context.t()
  def get_context(agent) do
    GenServer.call(agent, :get_context)
  end

  @doc """
  Dynamically loads a skill into the agent.
  """
  @spec load_skill(GenServer.server(), Skill.t()) :: :ok
  def load_skill(agent, %Skill{} = skill) do
    GenServer.cast(agent, {:load_skill, skill})
  end

  @doc """
  Resets the agent's conversation context.
  """
  @spec reset(GenServer.server()) :: :ok
  def reset(agent) do
    GenServer.cast(agent, :reset)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    state = %{
      id: opts[:id] || generate_id(),
      provider: Keyword.fetch!(opts, :provider),
      context: Context.new(),
      tools: opts[:tools] || [],
      skills: opts[:skills] || [],
      active_skill: nil,
      built_in_tools: opts[:built_in_tools] || [],
      status: :idle
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:chat, user_input, opts}, _from, state) do
    attachments = Keyword.get(opts, :files, [])
    {:ok, user_msg} = Message.new(role: :user, content: user_input, attachments: attachments)

    # Per-message built_in_tools override, or fall back to agent-level default
    chat_built_in_tools =
      Keyword.get(opts, :built_in_tools, state.built_in_tools)

    state = %{state | context: Context.add_message(state.context, user_msg), status: :processing}

    # Evaluate skills before calling LLM
    state = evaluate_and_apply_skills(state)

    case run_tool_loop(state, chat_built_in_tools, 0) do
      {:ok, response_msg, new_state} ->
        {:reply, {:ok, response_msg}, %{new_state | status: :idle}}

      {:handoff, target, context, new_state} ->
        {:reply, {:handoff, target, context}, %{new_state | status: :handed_off}}

      {:error, reason, new_state} ->
        {:reply, {:error, reason}, %{new_state | status: :idle}}
    end
  end

  @impl true
  def handle_call(:get_context, _from, state) do
    {:reply, state.context, state}
  end

  @impl true
  def handle_cast({:load_skill, skill}, state) do
    {:noreply, %{state | skills: state.skills ++ [skill]}}
  end

  @impl true
  def handle_cast(:reset, state) do
    {:noreply, %{state | context: Context.new(), active_skill: nil, status: :idle}}
  end

  @impl true
  def handle_cast({:receive_handoff, context}, state) do
    {:noreply, %{state | context: context, status: :idle}}
  end

  # Private Functions

  @spec run_tool_loop(state(), [atom()], non_neg_integer()) ::
          {:ok, Message.t(), state()}
          | {:handoff, pid() | atom(), Context.t(), state()}
          | {:error, term(), state()}
  defp run_tool_loop(state, _built_in_tools, iteration) when iteration >= @max_tool_iterations do
    {:error, :max_tool_iterations_reached, state}
  end

  defp run_tool_loop(state, built_in_tools, iteration) do
    effective_tools = get_effective_tools(state)
    messages = state.context.messages
    provider = %{state.provider | tools: effective_tools}

    case LlmProvider.chat(provider, messages, built_in_tools: built_in_tools) do
      {:ok, %Message{} = response_msg} ->
        new_context = Context.add_message(state.context, response_msg)
        {:ok, response_msg, %{state | context: new_context}}

      {:tool_call, name, args} ->
        # Record the assistant's tool call request in context
        {:ok, assistant_tc_msg} =
          Message.new(
            role: :assistant,
            content: "",
            tool_calls: [%{"name" => name, "args" => args}]
          )

        state = %{state | context: Context.add_message(state.context, assistant_tc_msg)}

        case execute_tool(name, args, effective_tools) do
          {:handoff, target, context} ->
            {:handoff, target, context, state}

          {:ok, result} ->
            {:ok, tool_msg} =
              Message.new(role: :tool, content: to_string(result), tool_call_id: name)

            new_context = Context.add_message(state.context, tool_msg)
            run_tool_loop(%{state | context: new_context}, built_in_tools, iteration + 1)

          {:error, reason} ->
            {:ok, error_msg} =
              Message.new(role: :tool, content: "Error: #{inspect(reason)}", tool_call_id: name)

            new_context = Context.add_message(state.context, error_msg)
            run_tool_loop(%{state | context: new_context}, built_in_tools, iteration + 1)
        end

      {:error, reason} ->
        {:error, reason, state}
    end
  end

  @spec execute_tool(String.t(), map(), [Tool.t()]) :: any()
  defp execute_tool(name, args, tools) do
    case Enum.find(tools, &(&1.name == name)) do
      nil -> {:error, "unknown tool: #{name}"}
      %Tool{function: fun} -> fun.(args)
    end
  end

  @spec get_effective_tools(state()) :: [Tool.t()]
  defp get_effective_tools(%{tools: tools, active_skill: nil}), do: tools

  defp get_effective_tools(%{tools: tools, active_skill: %Skill{tools: skill_tools}}) do
    tools ++ skill_tools
  end

  @spec evaluate_and_apply_skills(state()) :: state()
  defp evaluate_and_apply_skills(%{skills: []} = state), do: state

  defp evaluate_and_apply_skills(state) do
    case SkillsPattern.evaluate_skills(state.skills, state.context) do
      nil ->
        state

      %Skill{} = skill ->
        SkillsPattern.apply_skill(state, skill)
    end
  end

  @spec generate_id() :: String.t()
  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
