defmodule ExAgent do
  @moduledoc """
  Public API for the ExAgent multi-agent LLM library.

  ExAgent abstracts calls to various LLMs (OpenAI, Gemini, DeepSeek)
  via the `ExAgent.LlmProvider` protocol and orchestrates them using
  OTP primitives and four multi-agent design patterns.

  ## Quick Start

      # Create a provider
      provider = ExAgent.Providers.OpenAI.new(api_key: System.get_env("OPENAI_API_KEY"))

      # Start an agent
      {:ok, agent} = ExAgent.start_agent(provider: provider)

      # Chat
      {:ok, response} = ExAgent.chat(agent, "Hello!")

  ## Multi-Agent Patterns

  - **Subagents** - Centralized orchestration with isolated subagent calls
  - **Skills** - Progressive disclosure of specialized personas
  - **Handoffs** - State-driven transitions between agents
  - **Router** - Parallel dispatch and synthesis across agents
  """

  alias ExAgent.{Agent, Context, FileRef, FileUploader}
  alias ExAgent.Patterns.{Handoff, Router}

  # --- Agent Lifecycle ---

  @doc """
  Starts a new agent under the dynamic supervisor.

  ## Options

  - `:provider` (required) - struct implementing `ExAgent.LlmProvider`
  - `:id` - unique agent identifier
  - `:tools` - list of `ExAgent.Tool` structs
  - `:skills` - list of `ExAgent.Skill` structs
  - `:built_in_tools` - provider-specific built-in tools (e.g., `[:google_search]`, `[:web_search]`, `[:thinking]`)
  - `:name` - GenServer name for registration

  ## Examples

      provider = ExAgent.Providers.OpenAI.new(api_key: "sk-...")
      {:ok, pid} = ExAgent.start_agent(provider: provider)
  """
  @spec start_agent(Agent.agent_opts()) :: {:ok, pid()} | {:error, term()}
  defdelegate start_agent(opts), to: ExAgent.AgentDynamicSupervisor

  @doc """
  Stops an agent process.
  """
  @spec stop_agent(pid()) :: :ok | {:error, :not_found}
  defdelegate stop_agent(pid), to: ExAgent.AgentDynamicSupervisor

  # --- Chat ---

  @doc """
  Sends a user message to an agent and returns the response.

  ## Options

  - `:files` - list of file attachments, each a map with `:mime_type` and
    either `:data` (binary) or `:path` (file path). Files become part of
    the conversation context and are sent to the LLM alongside the text.
  - `:built_in_tools` - override agent-level built-in tools for this message.
    Examples: `[:google_search]`, `[:web_search]`, `[:thinking]`.

  ## Examples

      {:ok, response} = ExAgent.chat(agent, "What is Elixir?")
      response.content
      #=> "Elixir is a functional programming language..."

      # With file attachments
      {:ok, response} = ExAgent.chat(agent, "Describe this image",
        files: [%{path: "photo.jpg", mime_type: "image/jpeg"}])
  """
  @spec chat(GenServer.server(), String.t(), keyword()) ::
          {:ok, ExAgent.Message.t()}
          | {:handoff, pid() | atom(), Context.t()}
          | {:error, term()}
  defdelegate chat(agent, input, opts \\ []), to: Agent

  @doc """
  Sends a user message asynchronously, returning a Task.

  Accepts the same options as `chat/3`.
  """
  @spec chat_async(GenServer.server(), String.t(), keyword()) :: Task.t()
  defdelegate chat_async(agent, input, opts \\ []), to: Agent

  # --- Context ---

  @doc """
  Returns the agent's current conversation context.
  """
  @spec get_context(GenServer.server()) :: Context.t()
  defdelegate get_context(agent), to: Agent

  @doc """
  Reset conversation context.
  """
  @spec reset(GenServer.server()) :: :ok
  defdelegate reset(agent), to: Agent

  # --- File Uploads ---

  @doc """
  Uploads a file from disk to the provider and returns a reference.

  The returned `FileRef` can be passed in chat messages via
  `files: [%{file_ref: ref}]` to avoid sending base64-encoded data inline.

  ## Options

  - `:filename` - override filename (defaults to basename of `file_path`)
  - `:purpose` - OpenAI-specific file purpose (default: `"user_data"`)

  ## Examples

      provider = ExAgent.Providers.OpenAI.new(api_key: "sk-...")
      {:ok, ref} = ExAgent.upload_file(provider, "report.pdf", "application/pdf")
      {:ok, response} = ExAgent.chat(agent, "Summarize", files: [%{file_ref: ref}])
  """
  @spec upload_file(struct(), String.t(), String.t(), keyword()) ::
          {:ok, FileRef.t()} | {:error, term()}
  def upload_file(provider, file_path, mime_type, opts \\ []) do
    with {:ok, data} <- File.read(file_path) do
      opts = Keyword.put_new(opts, :filename, Path.basename(file_path))
      FileUploader.upload(provider, data, mime_type, opts)
    end
  end

  @doc """
  Uploads raw binary data to the provider and returns a reference.

  Use this when you already have file contents in memory.

  ## Options

  - `:filename` - filename for the upload (default: `"upload"`)
  - `:purpose` - OpenAI-specific file purpose (default: `"user_data"`)

  ## Examples

      image_bytes = File.read!("screenshot.png")
      {:ok, ref} = ExAgent.upload_data(provider, image_bytes, "image/png", filename: "screenshot.png")
  """
  @spec upload_data(struct(), binary(), String.t(), keyword()) ::
          {:ok, FileRef.t()} | {:error, term()}
  def upload_data(provider, data, mime_type, opts \\ []) do
    FileUploader.upload(provider, data, mime_type, opts)
  end

  # --- Patterns ---

  @doc """
  Routes input through matching agents and synthesizes results.

  See `ExAgent.Patterns.Router.route/2` for options.
  """
  @spec route(String.t(), Router.router_opts()) :: {:ok, String.t()} | {:error, term()}
  defdelegate route(input, opts), to: Router

  @doc """
  Transfers conversation context to a target agent.
  """
  @spec handoff(pid() | atom(), Context.t()) :: :ok
  defdelegate handoff(target, context), to: Handoff, as: :execute_handoff
end
