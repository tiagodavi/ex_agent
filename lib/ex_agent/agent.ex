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
          role: atom(),
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
          task_ref: reference() | nil,
          pre_turn_context: Context.t() | nil
        }

  @max_tool_iterations 10

  # Client API

  @doc """
  Starts an agent process linked to the current process.

  Takes either `:provider` (a struct) or `:role` (an atom resolved through
  `ExAgent.Roles`), never both.
  """
  @spec start_link(agent_opts()) :: GenServer.on_start()
  def start_link(opts) do
    opts = resolve_role(opts)
    name = opts[:name]
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  # Resolved here rather than in init/1 so both the direct call and the
  # DynamicSupervisor path (child_spec -> start_link) go through it once, and a
  # bad role raises at the caller instead of inside a supervised process.
  @spec resolve_role(agent_opts()) :: agent_opts()
  defp resolve_role(opts) do
    case {Keyword.fetch(opts, :role), Keyword.has_key?(opts, :provider)} do
      {{:ok, _role}, true} ->
        raise ArgumentError, "expected :provider or :role, not both"

      {{:ok, role}, false} ->
        opts |> Keyword.delete(:role) |> Keyword.put(:provider, ExAgent.Roles.fetch!(role))

      {:error, _has_provider?} ->
        opts
    end
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
  for the agent to return to idle.

  Never raises: a busy agent, a failed tool turn, and a mid-stream HTTP failure
  all arrive as a single `ExAgent.Chunk` with `type: :done` and an `:error`.

  ## Examples

      agent
      |> ExAgent.Agent.chat_stream("Explain OTP step by step")
      |> Enum.each(fn
        %ExAgent.Chunk{type: :text_delta, text: text} -> IO.write(text)
        _chunk -> :ok
      end)

  Or fold it back into the same response `chat/3` returns:

      {:ok, response} = agent |> ExAgent.Agent.chat_stream("Hi") |> ExAgent.collect()
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
            error_stream(:server, "tool resolution failed: #{inspect(reason)}")
        end

      {:error, :busy} ->
        error_stream(:invalid_request, "agent is processing another request")

      {:error, %ExAgent.Error{} = error} ->
        [ExAgent.Chunk.error(error)]
    end
  end

  # A one-chunk stream, so every failure mode reaches the consumer the same way.
  @spec error_stream(ExAgent.Error.type(), String.t()) :: [ExAgent.Chunk.t()]
  defp error_stream(type, message),
    do: [ExAgent.Chunk.error(ExAgent.Error.new(type, message))]

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
      task_ref: nil,
      pre_turn_context: nil
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:chat, _user_input, _opts}, _from, %{status: :processing} = state) do
    {:reply, {:error, :busy}, state}
  end

  def handle_call({:chat, user_input, opts}, from, state) do
    case build_user_message(user_input, opts) do
      {:error, error} ->
        {:reply, {:error, error}, state}

      {:ok, user_msg} ->
        # Per-message built_in_tools override, or fall back to agent-level default
        opts = Keyword.merge([built_in_tools: state.built_in_tools], opts)

        state = %{
          state
          | pre_turn_context: state.context,
            context: Context.add_message(state.context, user_msg),
            status: :processing
        }

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
  end

  @impl true
  def handle_call({:begin_stream, _input, _opts}, _from, %{status: :processing} = state) do
    {:reply, {:error, :busy}, state}
  end

  def handle_call({:begin_stream, user_input, opts}, _from, state) do
    case build_user_message(user_input, opts) do
      {:error, error} ->
        {:reply, {:error, error}, state}

      {:ok, user_msg} ->
        stream_opts = Keyword.merge([built_in_tools: state.built_in_tools], opts)

        state = %{
          state
          | pre_turn_context: state.context,
            context: Context.add_message(state.context, user_msg),
            status: :processing
        }

        state = evaluate_and_apply_skills(state)

        effective_tools = get_effective_tools(state)
        provider = with_tools(state.provider, effective_tools)

        {:reply, {:ok, provider, state.context.messages, effective_tools, stream_opts}, state}
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

  @impl true
  def handle_cast({:commit_stream, messages}, state) do
    context = Enum.reduce(messages, state.context, &Context.add_message(&2, &1))
    {:noreply, %{state | context: context, status: :idle}}
  end

  @impl true
  def handle_cast(:abort_stream, state) do
    {:noreply, %{state | context: rollback(state), status: :idle}}
  end

  @impl true
  def handle_info({ref, result}, %{task_ref: ref} = state) do
    Process.demonitor(ref, [:flush])
    {reply, new_state} = handle_loop_result(result, state)
    GenServer.reply(state.reply_to, reply)
    {:noreply, %{new_state | reply_to: nil, task_ref: nil}}
  end

  def handle_info({:DOWN, ref, :process, _pid, reason}, %{task_ref: ref} = state) do
    error = %{
      ExAgent.Error.new(:server, "agent task failed: #{inspect(reason)}")
      | raw: reason
    }

    GenServer.reply(state.reply_to, {:error, error})
    {:noreply, %{state | status: :idle, reply_to: nil, task_ref: nil}}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # Private Functions

  # A malformed attachment must not crash the agent mid-handle_call — it is
  # caller input, so it comes back as an error like any other bad request.
  @spec build_user_message(String.t(), keyword()) ::
          {:ok, Message.t()} | {:error, ExAgent.Error.t()}
  defp build_user_message(user_input, opts) do
    attachments = Keyword.get(opts, :files, [])

    case Message.new(role: :user, content: user_input, attachments: attachments) do
      {:ok, message} -> {:ok, message}
      {:error, reason} -> {:error, ExAgent.Error.new(:invalid_request, reason)}
    end
  end

  @spec handle_loop_result(term(), state()) :: {term(), state()}
  defp handle_loop_result({:ok, response, loop_state}, state) do
    {{:ok, response}, %{state | context: loop_state.context, status: :idle}}
  end

  defp handle_loop_result({:handoff, target, context, loop_state}, state) do
    {{:handoff, target, context}, %{state | context: loop_state.context, status: :handed_off}}
  end

  # A turn either completes or it did not happen. Committing a failed turn left a
  # message the provider had already refused — a rejected attachment, say — in
  # history to be resent on every later turn, breaking the agent permanently; and
  # it duplicated the question when the caller retried a transient failure.
  defp handle_loop_result({:error, reason, _loop_state}, state) do
    {{:error, reason}, %{state | context: rollback(state), status: :idle}}
  end

  @spec rollback(state()) :: Context.t()
  defp rollback(%{pre_turn_context: nil, context: context}), do: context
  defp rollback(%{pre_turn_context: context}), do: context

  @spec run_tool_loop(state(), [atom()], non_neg_integer()) ::
          {:ok, ExAgent.Response.t(), state()}
          | {:handoff, pid() | atom(), Context.t(), state()}
          | {:error, term(), state()}
  defp run_tool_loop(state, _opts, iteration) when iteration >= @max_tool_iterations do
    {:error, :max_tool_iterations_reached, state}
  end

  defp run_tool_loop(state, opts, iteration) do
    effective_tools = get_effective_tools(state)
    messages = state.context.messages
    provider = with_tools(state.provider, effective_tools)

    case Provider.chat(provider, messages, opts) do
      {:ok, %ExAgent.Response{} = response} ->
        new_context = Context.add_message(state.context, response.message)
        {:ok, response, %{state | context: new_context}}

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
      {:ok, %ExAgent.Response{}} ->
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
    case open_stream(provider, messages, opts) do
      {:ok, stream} ->
        commit_on_completion(agent, stream, pending)

      {:error, error} ->
        # Refused before any request was made. Release the agent and drop the
        # turn, so one bad attachment does not poison every later one.
        GenServer.cast(agent, :abort_stream)
        [ExAgent.Chunk.error(error)]
    end
  end

  # `Provider.stream/3` raises rather than returning a tuple, because a lazy
  # enumerable has nowhere to carry an error at construction time. `chat_stream/3`
  # promises never to raise, so the agent converts it here.
  @spec open_stream(struct(), [Message.t()], keyword()) ::
          {:ok, Enumerable.t()} | {:error, ExAgent.Error.t()}
  defp open_stream(provider, messages, opts) do
    {:ok, Provider.stream(provider, messages, opts)}
  rescue
    error in ExAgent.Error -> {:error, error}
  end

  @spec commit_on_completion(GenServer.server(), Enumerable.t(), [Message.t()]) :: Enumerable.t()
  defp commit_on_completion(agent, stream, pending) do
    stream
    |> Stream.transform(
      fn -> {[], []} end,
      fn chunk, acc -> {[chunk], accumulate(chunk, acc)} end,
      fn {texts, calls} ->
        # The after-fun also runs on halt, so a failed or abandoned stream still
        # releases the agent; leaving it :processing forever is worse than
        # committing a partial turn, and the user message is already in context.
        {:ok, final} =
          Message.new(
            role: :assistant,
            content: texts |> Enum.reverse() |> IO.iodata_to_binary(),
            tool_calls: calls |> Enum.reverse() |> ExAgent.Chunk.finalize_tool_calls()
          )

        GenServer.cast(agent, {:commit_stream, pending ++ [final]})
        :ok
      end
    )
  end

  # Reasoning traces are deliberately excluded: they are not conversation
  # history, and replaying them corrupts the next turn.
  @spec accumulate(ExAgent.Chunk.t(), {[String.t()], [ExAgent.Chunk.t()]}) ::
          {[String.t()], [ExAgent.Chunk.t()]}
  defp accumulate(%ExAgent.Chunk{type: :text_delta, text: text}, {texts, calls})
       when is_binary(text),
       do: {[text | texts], calls}

  defp accumulate(%ExAgent.Chunk{type: :tool_call_delta} = chunk, {texts, calls}),
    do: {texts, [chunk | calls]}

  defp accumulate(_chunk, acc), do: acc

  @spec execute_tool(String.t(), map(), [Tool.t()]) :: any()
  defp execute_tool(name, args, tools) do
    case Enum.find(tools, &(&1.name == name)) do
      nil ->
        {:error, "unknown tool: #{name}"}

      %Tool{function: fun} ->
        normalize_tool_result(fun.(args))
    end
  end

  # `ExAgent.Tool`'s :function is typed `(map() -> any())`, so a tool is free to
  # return a bare value. Treat that as the result instead of crashing the loop.
  @spec normalize_tool_result(term()) ::
          {:ok, term()} | {:error, term()} | {:handoff, pid() | atom(), Context.t()}
  defp normalize_tool_result({:ok, result}), do: {:ok, result}
  defp normalize_tool_result({:error, reason}), do: {:error, reason}
  defp normalize_tool_result({:handoff, _target, _context} = handoff), do: handoff
  defp normalize_tool_result(other), do: {:ok, other}

  # Services read tools off the provider struct, so the agent populates the field
  # — but only when the provider declares one. A provider with no tool support
  # has no reason to carry it, and a KeyError surfacing as an opaque :server
  # error is a poor way to say so.
  @spec with_tools(struct(), [Tool.t()]) :: struct()
  defp with_tools(provider, tools) do
    if Map.has_key?(provider, :tools), do: %{provider | tools: tools}, else: provider
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
