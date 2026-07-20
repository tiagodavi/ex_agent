defmodule ExAgent.Agent do
  @moduledoc """
  GenServer that manages a single LLM agent.

  Holds the provider struct in state and dispatches calls via the
  `ExAgent.Provider` behaviour. Contains the tool execution loop
  that automatically invokes tools when the LLM requests them.

  ## State

  The agent state includes:
  - `provider` - any struct implementing `ExAgent.Provider`
  - `context` - conversation history (`ExAgent.Context`)
  - `tools` - available tools for function-calling
  - `skills` - loadable skill definitions
  - `active_skill` - currently active skill (if any)
  """

  use GenServer

  alias ExAgent.{Context, Message, Provider, Skill, Tool}
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
          status: :idle | :processing | :handed_off,
          reply_to: GenServer.from() | nil,
          task_ref: reference() | nil
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
  Sends a user message and returns a lazy `Stream` of the assistant's text chunks.

  Tool-call turns (if any tools are configured) are resolved first without
  streaming; only the final assistant turn is streamed. When no tools are
  configured the response is streamed directly.

  The conversation context is committed when the stream is fully consumed, so
  **the returned stream must be consumed** (e.g. with `Enum`/`Stream` functions)
  for the agent to return to idle. Raises `ExAgent.StreamError` if the agent is
  busy with another request.

  ## Examples

      agent
      |> ExAgent.Agent.chat_stream("Explain OTP step by step")
      |> Enum.each(&IO.write/1)
  """
  @spec chat_stream(GenServer.server(), String.t(), keyword()) :: Enumerable.t()
  def chat_stream(agent, user_input, opts \\ []) when is_binary(user_input) do
    case GenServer.call(agent, {:begin_stream, user_input, opts}, :infinity) do
      {:ok, provider, messages, [], stream_opts} ->
        committing_stream(agent, provider, messages, stream_opts, [])

      {:ok, provider, messages, _tools, stream_opts} ->
        case resolve_tools(provider, messages, stream_opts, [], 0) do
          {:ok, resolved_messages, pending} ->
            committing_stream(agent, provider, resolved_messages, stream_opts, pending)

          {:error, reason} ->
            GenServer.cast(agent, {:commit_stream, []})
            raise ExAgent.StreamError, status: nil, body: reason
        end

      {:error, :busy} ->
        raise ExAgent.StreamError, status: nil, body: :agent_busy
    end
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
      status: :idle,
      reply_to: nil,
      task_ref: nil
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:chat, _user_input, _opts}, _from, %{status: :processing} = state) do
    {:reply, {:error, :busy}, state}
  end

  def handle_call({:chat, user_input, opts}, from, state) do
    attachments = Keyword.get(opts, :files, [])
    {:ok, user_msg} = Message.new(role: :user, content: user_input, attachments: attachments)

    # Per-message built_in_tools override, or fall back to agent-level default
    opts = Keyword.merge([built_in_tools: state.built_in_tools], opts)

    state = %{state | context: Context.add_message(state.context, user_msg), status: :processing}

    # Evaluate skills before calling LLM
    state = evaluate_and_apply_skills(state)

    # Run the (potentially long, IO-bound) tool loop off the GenServer so the
    # process stays responsive to reads (get_context/status) and casts while it
    # works. The caller stays blocked until we GenServer.reply/2 in handle_info.
    task =
      Task.Supervisor.async_nolink(ExAgent.TaskSupervisor, fn ->
        run_tool_loop(state, opts, 0)
      end)

    {:noreply, %{state | reply_to: from, task_ref: task.ref}}
  end

  @impl true
  def handle_call({:begin_stream, _input, _opts}, _from, %{status: :processing} = state) do
    {:reply, {:error, :busy}, state}
  end

  def handle_call({:begin_stream, user_input, opts}, _from, state) do
    attachments = Keyword.get(opts, :files, [])
    {:ok, user_msg} = Message.new(role: :user, content: user_input, attachments: attachments)

    stream_opts = Keyword.merge([built_in_tools: state.built_in_tools], opts)

    state = %{state | context: Context.add_message(state.context, user_msg), status: :processing}
    state = evaluate_and_apply_skills(state)

    effective_tools = get_effective_tools(state)
    provider = %{state.provider | tools: effective_tools}

    {:reply, {:ok, provider, state.context.messages, effective_tools, stream_opts}, state}
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

  @impl true
  def handle_cast({:commit_stream, messages}, state) do
    context = Enum.reduce(messages, state.context, &Context.add_message(&2, &1))
    {:noreply, %{state | context: context, status: :idle}}
  end

  @impl true
  def handle_info({ref, result}, %{task_ref: ref} = state) do
    Process.demonitor(ref, [:flush])
    {reply, new_state} = handle_loop_result(result, state)
    GenServer.reply(state.reply_to, reply)
    {:noreply, %{new_state | reply_to: nil, task_ref: nil}}
  end

  def handle_info({:DOWN, ref, :process, _pid, reason}, %{task_ref: ref} = state) do
    GenServer.reply(state.reply_to, {:error, {:agent_task_failed, reason}})
    {:noreply, %{state | status: :idle, reply_to: nil, task_ref: nil}}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # Private Functions

  @spec handle_loop_result(term(), state()) :: {term(), state()}
  defp handle_loop_result({:ok, response_msg, loop_state}, state) do
    {{:ok, response_msg}, %{state | context: loop_state.context, status: :idle}}
  end

  defp handle_loop_result({:handoff, target, context, loop_state}, state) do
    {{:handoff, target, context}, %{state | context: loop_state.context, status: :handed_off}}
  end

  defp handle_loop_result({:error, reason, loop_state}, state) do
    {{:error, reason}, %{state | context: loop_state.context, status: :idle}}
  end

  @spec run_tool_loop(state(), [atom()], non_neg_integer()) ::
          {:ok, Message.t(), state()}
          | {:handoff, pid() | atom(), Context.t(), state()}
          | {:error, term(), state()}
  defp run_tool_loop(state, _opts, iteration) when iteration >= @max_tool_iterations do
    {:error, :max_tool_iterations_reached, state}
  end

  defp run_tool_loop(state, opts, iteration) do
    effective_tools = get_effective_tools(state)
    messages = state.context.messages
    provider = %{state.provider | tools: effective_tools}

    case Provider.chat(provider, messages, opts) do
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
            run_tool_loop(%{state | context: new_context}, opts, iteration + 1)

          {:error, reason} ->
            {:ok, error_msg} =
              Message.new(role: :tool, content: "Error: #{inspect(reason)}", tool_call_id: name)

            new_context = Context.add_message(state.context, error_msg)
            run_tool_loop(%{state | context: new_context}, opts, iteration + 1)
        end

      {:error, reason} ->
        {:error, reason, state}
    end
  end

  # Resolves tool-call turns (non-streamed) before the final streamed turn.
  # Returns the message list to stream plus the intermediate messages to commit.
  @spec resolve_tools(struct(), [Message.t()], keyword(), [Message.t()], non_neg_integer()) ::
          {:ok, [Message.t()], [Message.t()]} | {:error, term()}
  defp resolve_tools(_provider, _messages, _opts, _pending, iteration)
       when iteration >= @max_tool_iterations do
    {:error, :max_tool_iterations_reached}
  end

  defp resolve_tools(provider, messages, opts, pending, iteration) do
    case Provider.chat(provider, messages, opts) do
      {:ok, %Message{}} ->
        # Final turn: discard its text; stream a fresh generation from `messages`.
        {:ok, messages, pending}

      {:tool_call, name, args} ->
        {:ok, tc_msg} =
          Message.new(
            role: :assistant,
            content: "",
            tool_calls: [%{"name" => name, "args" => args}]
          )

        result_msg = execute_tool_message(name, args, provider.tools)

        case result_msg do
          {:error, reason} ->
            {:error, reason}

          %Message{} = tool_msg ->
            new_messages = messages ++ [tc_msg, tool_msg]

            resolve_tools(
              provider,
              new_messages,
              opts,
              pending ++ [tc_msg, tool_msg],
              iteration + 1
            )
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec execute_tool_message(String.t(), map(), [Tool.t()]) :: Message.t() | {:error, term()}
  defp execute_tool_message(name, args, tools) do
    case execute_tool(name, args, tools) do
      {:handoff, _target, _context} ->
        {:error, :handoff_not_supported_in_stream}

      {:ok, result} ->
        {:ok, msg} = Message.new(role: :tool, content: to_string(result), tool_call_id: name)
        msg

      {:error, reason} ->
        {:ok, msg} =
          Message.new(role: :tool, content: "Error: #{inspect(reason)}", tool_call_id: name)

        msg
    end
  end

  # Wraps the provider stream so the final assistant message (and any pending
  # tool messages) are committed to the agent's context once fully consumed.
  @spec committing_stream(GenServer.server(), struct(), [Message.t()], keyword(), [Message.t()]) ::
          Enumerable.t()
  defp committing_stream(agent, provider, messages, opts, pending) do
    provider
    |> Provider.stream(messages, opts)
    |> Stream.transform(
      fn -> "" end,
      fn chunk, acc -> {[chunk], acc <> chunk} end,
      fn acc ->
        {:ok, final} = Message.new(role: :assistant, content: acc)
        GenServer.cast(agent, {:commit_stream, pending ++ [final]})
        :ok
      end
    )
  end

  @spec execute_tool(String.t(), map(), [Tool.t()]) :: any()
  defp execute_tool(name, args, tools) do
    case Enum.find(tools, &(&1.name == name)) do
      nil ->
        {:error, "unknown tool: #{name}"}

      %Tool{function: fun} ->
        fun.(args)
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
