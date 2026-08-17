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

  ## Options beyond the provider

  - `:max_tool_iterations` - ceiling on tool round-trips (default 10)
  - `:max_history` - keep at most this many messages; unbounded when unset
  - `:retain_attachments` - send attachments from earlier turns again on every
    request (default `true`)
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
          max_tool_iterations: pos_integer(),
          max_history: pos_integer(),
          retain_attachments: boolean(),
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
          max_tool_iterations: pos_integer(),
          max_history: pos_integer() | nil,
          retain_attachments: boolean(),
          base_system_prompt: String.t() | nil,
          status: :idle | :processing | :handed_off,
          reply_to: GenServer.from() | nil,
          task_ref: reference() | nil,
          pre_turn_context: Context.t() | nil
        }

  # A ceiling on tool round-trips, so a model that keeps calling tools cannot
  # bill forever. Overridable per agent with `:max_tool_iterations`.
  @default_max_tool_iterations 10

  # Client API

  @doc """
  Starts an agent process linked to the current process.

  Takes either `:provider` (a struct) or `:role` (an atom resolved through
  `ExAgent.Roles`), never both.
  """
  @spec start_link(agent_opts()) :: GenServer.on_start()
  def start_link(opts) do
    opts = opts |> resolve_role() |> validate_positive!(:max_tool_iterations)
    opts = validate_positive!(opts, :max_history)
    opts = validate_boolean!(opts, :retain_attachments)
    name = opts[:name]
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  # A bad ceiling used to be accepted here and then crash the agent at the end of
  # its first turn, which points at the wrong line entirely.
  # Only `true`/`false` mean anything here, and `nil` is not "use the default":
  # omitting the key is. Accepting nil would make a typo'd value look deliberate.
  @spec validate_boolean!(agent_opts(), atom()) :: agent_opts()
  defp validate_boolean!(opts, key) do
    case Keyword.fetch(opts, key) do
      :error ->
        opts

      {:ok, value} when is_boolean(value) ->
        opts

      {:ok, value} ->
        raise ArgumentError, "#{inspect(key)} must be true or false, got #{inspect(value)}"
    end
  end

  @spec validate_positive!(agent_opts(), atom()) :: agent_opts()
  defp validate_positive!(opts, key) do
    case Keyword.get(opts, key) do
      nil ->
        opts

      value when is_integer(value) and value > 0 ->
        opts

      value ->
        raise ArgumentError, ":#{key} must be a positive integer, got #{inspect(value)}"
    end
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
          {:ok, ExAgent.Response.t()}
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

  Every turn is streamed, including one that requests tools: its tool calls are
  executed and the next turn opened, so a multi-turn answer costs one completion
  per turn and still ends with exactly one `:done` chunk.

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
      {:ok, provider, messages, tools, stream_opts} ->
        turn_stream(agent, provider, messages, stream_opts, tools)

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
      max_tool_iterations: opts[:max_tool_iterations] || @default_max_tool_iterations,
      max_history: opts[:max_history],
      retain_attachments: Keyword.get(opts, :retain_attachments, true),
      # Remembered so a skill's prompt can be undone; a skill that activates once
      # must not overwrite the agent's own persona for the rest of its life.
      base_system_prompt: Map.get(Keyword.fetch!(opts, :provider), :system_prompt),
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

        {:reply, {:ok, provider, outbound_messages(state), effective_tools, stream_opts}, state}
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
    {:noreply, %{state | context: trim(state, context), status: :idle}}
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

  # A malformed attachment must not crash the agent mid-handle_call - it is
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
    {{:ok, response}, %{state | context: trim(state, loop_state.context), status: :idle}}
  end

  defp handle_loop_result({:handoff, target, context, loop_state}, state) do
    {{:handoff, target, context}, %{state | context: loop_state.context, status: :handed_off}}
  end

  # A turn either completes or it did not happen. Committing a failed turn left a
  # message the provider had already refused - a rejected attachment, say - in
  # history to be resent on every later turn, breaking the agent permanently; and
  # it duplicated the question when the caller retried a transient failure.
  defp handle_loop_result({:error, reason, _loop_state}, state) do
    {{:error, reason}, %{state | context: rollback(state), status: :idle}}
  end

  # Bounded history is opt-in: dropping what a user said is the caller's call.
  # The messages actually sent, which is not the same thing as the messages kept.
  #
  # Provider APIs are stateless, so the whole history goes out every turn and
  # every earlier attachment goes with it, re-encoded. Media caps are per
  # *request*, so an agent sending one image per turn hits a per-request limit it
  # never knowingly exceeded. With `retain_attachments: false` only the newest
  # user message keeps its attachments; history itself is untouched, so
  # `get_context/1` still records what was sent.
  @spec outbound_messages(state()) :: [Message.t()]
  defp outbound_messages(%{retain_attachments: true, context: context}), do: context.messages

  defp outbound_messages(%{context: context}) do
    messages = context.messages
    keep = last_attachment_index(messages)

    messages
    |> Enum.with_index()
    |> Enum.map(fn
      {message, ^keep} -> message
      {%Message{attachments: []} = message, _index} -> message
      {message, _index} -> %{message | attachments: []}
    end)
  end

  # The current turn is the newest user message; anything before it is history.
  @spec last_attachment_index([Message.t()]) :: integer()
  defp last_attachment_index(messages) do
    messages
    |> Enum.with_index()
    |> Enum.reduce(-1, fn
      {%Message{role: :user}, index}, _acc -> index
      {_message, _index}, acc -> acc
    end)
  end

  @spec trim(state(), Context.t()) :: Context.t()
  defp trim(%{max_history: nil}, context), do: context
  defp trim(%{max_history: max}, context), do: Context.trim(context, max)

  @spec rollback(state()) :: Context.t()
  defp rollback(%{pre_turn_context: nil, context: context}), do: context
  defp rollback(%{pre_turn_context: context}), do: context

  @spec run_tool_loop(state(), [atom()], non_neg_integer()) ::
          {:ok, ExAgent.Response.t(), state()}
          | {:handoff, pid() | atom(), Context.t(), state()}
          | {:error, term(), state()}
  defp run_tool_loop(%{max_tool_iterations: max} = state, _opts, iteration)
       when iteration >= max do
    {:error, :max_tool_iterations_reached, state}
  end

  defp run_tool_loop(state, opts, iteration) do
    effective_tools = get_effective_tools(state)
    messages = outbound_messages(state)
    provider = with_tools(state.provider, effective_tools)

    case Provider.chat(provider, messages, opts) do
      {:ok, %ExAgent.Response{} = response} ->
        new_context = Context.add_message(state.context, response.message)
        {:ok, response, %{state | context: new_context}}

      {:error, reason} ->
        {:error, reason, state}

      tool_request ->
        run_tool_calls(normalize_calls(tool_request), state, opts, iteration, effective_tools)
    end
  end

  # A turn can request several tools at once. Each result is appended before the
  # next call runs, so the provider sees one response per call - a model that
  # asked for three and got one back would re-request the missing two forever.
  @spec run_tool_calls([map()], state(), keyword(), non_neg_integer(), [Tool.t()]) ::
          {:ok, ExAgent.Response.t(), state()}
          | {:handoff, pid() | atom(), Context.t(), state()}
          | {:error, term(), state()}
  defp run_tool_calls(calls, state, opts, iteration, tools) do
    {:ok, assistant_msg} = Message.new(role: :assistant, content: "", tool_calls: calls)
    state = %{state | context: Context.add_message(state.context, assistant_msg)}

    case Enum.reduce_while(calls, {:cont, state}, &apply_call(&1, &2, tools)) do
      {:cont, state} -> run_tool_loop(state, opts, iteration + 1)
      {:handoff, target, context, state} -> {:handoff, target, context, state}
    end
  end

  @spec apply_call(map(), {:cont, state()}, [Tool.t()]) :: {:cont | :halt, term()}
  defp apply_call(call, {:cont, state}, tools) do
    name = call["name"]

    case execute_tool(name, call["args"] || %{}, tools) do
      {:handoff, target, context} ->
        {:halt, {:handoff, target, context, state}}

      {:ok, result} ->
        {:cont, {:cont, record_result(state, call, stringify(result))}}

      {:error, reason} ->
        {:cont, {:cont, record_result(state, call, "Error: #{inspect(reason)}")}}
    end
  end

  # The tool name travels alongside the id: OpenAI correlates a result by id,
  # Gemini by function name, and only one of the two can be the `tool_call_id`.
  @spec record_result(state(), map(), String.t()) :: state()
  defp record_result(state, call, content) do
    {:ok, message} =
      Message.new(
        role: :tool,
        content: content,
        tool_call_id: call["id"] || call["name"],
        metadata: %{tool_name: call["name"]}
      )

    %{state | context: Context.add_message(state.context, message)}
  end

  # Both shapes of the `c:ExAgent.Provider.chat/3` contract, normalized.
  @spec normalize_calls(term()) :: [map()]
  defp normalize_calls({:tool_calls, calls}) when is_list(calls), do: calls

  defp normalize_calls({:tool_call, name, args}),
    do: [%{"id" => name, "name" => name, "args" => args}]

  # A tool is typed `(map() -> any())` and JSON-shaped results are the norm, so
  # anything that is not already text is encoded rather than crashing the turn.
  @spec stringify(term()) :: String.t()
  defp stringify(result) when is_binary(result), do: result
  defp stringify(result) when is_atom(result) or is_number(result), do: to_string(result)

  defp stringify(result) do
    case Jason.encode(result) do
      {:ok, json} -> json
      {:error, _reason} -> inspect(result)
    end
  end

  # Streams every turn, resolving tool calls between them.
  #
  # The previous shape ran the whole tool loop *non-streamed* and then threw the
  # finished answer away to regenerate it as a stream - two full completions for
  # every streamed turn with tools configured. Here each turn is streamed once:
  # a turn that ends in tool calls has its `:done` swallowed, its tools run, and
  # the next turn opened, so the consumer still sees exactly one terminal chunk.
  @spec turn_stream(GenServer.server(), struct(), [Message.t()], keyword(), [Tool.t()]) ::
          Enumerable.t()
  defp turn_stream(agent, provider, messages, opts, tools) do
    Stream.resource(
      fn -> open_turn(provider, messages, opts, new_turn_state(messages)) end,
      &pull(&1, agent, provider, opts, tools),
      &finish_stream(&1, agent)
    )
  end

  @spec new_turn_state([Message.t()]) :: map()
  defp new_turn_state(messages) do
    %{cont: nil, messages: messages, pending: [], texts: [], calls: [], iteration: 0}
  end

  # Opens one turn, converting `Provider.stream/3`'s raise into a terminal chunk:
  # `chat_stream/3` promises never to raise, and a lazy enumerable has nowhere to
  # carry an error tuple at construction time.
  @spec open_turn(struct(), [Message.t()], keyword(), map()) :: map()
  defp open_turn(provider, messages, opts, state) do
    stream = Provider.stream(provider, messages, opts)
    %{state | cont: suspendable(stream), messages: messages}
  rescue
    error in ExAgent.Error -> Map.put(state, :abort, error)
  end

  # A pull-based view of an enumerable, so one turn can be consumed lazily and
  # the next opened only if the model asked for tools.
  @spec suspendable(Enumerable.t()) :: (term() -> term())
  defp suspendable(stream) do
    &Enumerable.reduce(stream, &1, fn chunk, _acc -> {:suspend, chunk} end)
  end

  @spec pull(map(), GenServer.server(), struct(), keyword(), [Tool.t()]) ::
          {[ExAgent.Chunk.t()], map()} | {:halt, map()}
  # Halts with the state, not a sentinel, so the after-fun still sees the turn
  # it has to commit.
  defp pull(%{released: true} = state, _agent, _provider, _opts, _tools), do: {:halt, state}

  defp pull(%{abort: error} = state, agent, _provider, _opts, _tools) do
    # Refused before any request was made. Release the agent and drop the turn,
    # so one bad attachment does not poison every later one.
    GenServer.cast(agent, :abort_stream)
    {[ExAgent.Chunk.error(error)], Map.put(state, :released, true)}
  end

  defp pull(state, _agent, provider, opts, tools) do
    case state.cont.({:cont, nil}) do
      {:suspended, chunk, cont} ->
        handle_chunk(chunk, %{state | cont: cont}, provider, opts, tools)

      # A provider stream always ends with `:done`; reaching the end without one
      # still has to terminate the consumer's stream.
      _done_or_halted ->
        {[ExAgent.Chunk.done()], Map.put(state, :released, true)}
    end
  end

  @spec handle_chunk(ExAgent.Chunk.t(), map(), struct(), keyword(), [Tool.t()]) ::
          {[ExAgent.Chunk.t()], map()}
  defp handle_chunk(%ExAgent.Chunk{type: :done} = chunk, state, provider, opts, tools) do
    calls = ExAgent.Chunk.finalize_tool_calls(Enum.reverse(state.calls)) || []

    if continue?(chunk, calls, tools, state) do
      continue_turn(calls, state, provider, opts, tools)
    else
      # The terminal chunk: emit it, then halt on the next pull so the stream
      # ends with exactly one `:done`.
      {[chunk], Map.put(state, :released, true)}
    end
  end

  defp handle_chunk(chunk, state, _provider, _opts, _tools),
    do: {[chunk], accumulate(chunk, state)}

  # Only a turn that actually requested tools the agent can run continues; a
  # `:tool_calls` finish reason with no executable call would loop forever.
  @spec continue?(ExAgent.Chunk.t(), [map()], [Tool.t()], map()) :: boolean()
  defp continue?(%ExAgent.Chunk{finish_reason: :error}, _calls, _tools, _state), do: false

  defp continue?(_chunk, calls, tools, state) do
    calls != [] and tools != [] and state.iteration < @default_max_tool_iterations
  end

  @spec continue_turn([map()], map(), struct(), keyword(), [Tool.t()]) ::
          {[ExAgent.Chunk.t()], map()}
  defp continue_turn(calls, state, provider, opts, tools) do
    {:ok, assistant} =
      Message.new(role: :assistant, content: turn_text(state), tool_calls: calls)

    case tool_messages(calls, tools) do
      {:error, reason} ->
        {[ExAgent.Chunk.error(ExAgent.Error.new(:server, "tool failed: #{inspect(reason)}"))],
         state}

      tool_msgs ->
        turn = [assistant | tool_msgs]

        state = %{
          state
          | pending: state.pending ++ turn,
            texts: [],
            calls: [],
            iteration: state.iteration + 1
        }

        {[], open_turn(provider, state.messages ++ turn, opts, state)}
    end
  end

  @spec turn_text(map()) :: String.t()
  defp turn_text(%{texts: texts}), do: texts |> Enum.reverse() |> IO.iodata_to_binary()

  # Runs on halt too, so an abandoned stream still releases the agent; leaving it
  # `:processing` forever is worse than committing a partial turn, and the user
  # message is already in context.
  @spec finish_stream(map(), GenServer.server()) :: :ok
  defp finish_stream(%{abort: _error}, _agent), do: :ok

  defp finish_stream(state, agent) do
    {:ok, final} =
      Message.new(
        role: :assistant,
        content: turn_text(state),
        tool_calls: ExAgent.Chunk.finalize_tool_calls(Enum.reverse(state.calls))
      )

    GenServer.cast(agent, {:commit_stream, state.pending ++ [final]})
    :ok
  end

  @spec tool_messages([map()], [Tool.t()]) :: [Message.t()] | {:error, term()}
  defp tool_messages(calls, tools) do
    Enum.reduce_while(calls, [], fn call, acc ->
      case execute_tool(call["name"], call["args"] || %{}, tools) do
        {:handoff, _target, _context} ->
          {:halt, {:error, :handoff_not_supported_in_stream}}

        {:ok, result} ->
          {:cont, acc ++ [tool_message(call, stringify(result))]}

        {:error, reason} ->
          {:cont, acc ++ [tool_message(call, "Error: #{inspect(reason)}")]}
      end
    end)
  end

  @spec tool_message(map(), String.t()) :: Message.t()
  defp tool_message(call, content) do
    {:ok, message} =
      Message.new(
        role: :tool,
        content: content,
        tool_call_id: call["id"] || call["name"],
        metadata: %{tool_name: call["name"]}
      )

    message
  end

  # Reasoning traces are deliberately excluded: they are not conversation
  # history, and replaying them corrupts the next turn.
  @spec accumulate(ExAgent.Chunk.t(), map()) :: map()
  defp accumulate(%ExAgent.Chunk{type: :text_delta, text: text}, state) when is_binary(text),
    do: %{state | texts: [text | state.texts]}

  defp accumulate(%ExAgent.Chunk{type: :tool_call_delta} = chunk, state),
    do: %{state | calls: [chunk | state.calls]}

  defp accumulate(_chunk, state), do: state

  @spec execute_tool(String.t(), map(), [Tool.t()]) :: any()
  defp execute_tool(name, args, tools) do
    case Enum.find(tools, &(&1.name == name)) do
      nil ->
        {:error, "unknown tool: #{name}"}

      %Tool{} = tool ->
        ExAgent.Telemetry.span([:tool], %{tool: name}, fn ->
          case Tool.cast_args(tool, args) do
            {:ok, cast} -> normalize_tool_result(tool.function.(cast))
            # Feedback, not a crash: the model can fix its own arguments.
            {:error, message} -> {:error, message}
          end
        end)
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
  # - but only when the provider declares one. A provider with no tool support
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

  # Skills are re-evaluated every turn, so deactivation has to be handled too:
  # applying one used to overwrite the provider's system prompt permanently,
  # leaving a "SQL expert" answering jokes for the rest of the process's life.
  @spec evaluate_and_apply_skills(state()) :: state()
  defp evaluate_and_apply_skills(%{skills: []} = state), do: state

  defp evaluate_and_apply_skills(state) do
    case SkillsPattern.evaluate(state.skills, state.context) do
      nil -> SkillsPattern.clear_skill(state)
      %Skill{} = skill -> SkillsPattern.apply_skill(state, skill)
    end
  end

  @spec generate_id() :: String.t()
  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
