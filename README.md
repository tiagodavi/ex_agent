# ExAgent

An Elixir library for building multi-agent LLM applications. ExAgent abstracts calls to various LLM providers (OpenAI, Gemini, DeepSeek) via an extensible Protocol and orchestrates them using OTP primitives with four multi-agent design patterns.

## Features

- **Protocol-based LLM abstraction** — Swap providers without changing application code
- **Built on OTP** — Agents backed by GenServers, supervised processes, async Tasks
- **Automatic tool execution** — Define tools once, the agent loops LLM calls until complete
- **4 multi-agent patterns** — Subagents, Skills, Handoffs, Router
- **HTTP via Req** — Clean, composable HTTP with built-in JSON encoding and auth
- **Multimodal file attachments** — Send images, PDFs, and other files alongside chat messages
- **Extensible** — Add any LLM provider by implementing a single protocol

## Installation

Add `ex_agent` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ex_agent, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
# 1. Create a provider
provider = ExAgent.Providers.OpenAI.new(api_key: System.get_env("OPENAI_API_KEY"))

# 2. Start an agent
{:ok, agent} = ExAgent.start_agent(provider: provider)

# 3. Chat
{:ok, response} = ExAgent.chat(agent, "What is Elixir?")
IO.puts(response.content)
```

## Providers

ExAgent ships with three built-in providers. Each is configured via `new/1` and automatically initializes a Req HTTP client.

### OpenAI

Supports chat and file attachments (images via `image_url` multipart format).

```elixir
provider = ExAgent.Providers.OpenAI.new(
  api_key: "sk-...", # required
  model: "gpt-4o", # default: "gpt-4o"
  base_url: "https://api.openai.com/v1",  # default
  system_prompt: "You are a helpful assistant."
)
```

### Gemini

Supports chat and file attachments (images, PDFs, etc. via `inline_data` format).

```elixir
provider = ExAgent.Providers.Gemini.new(
  api_key: "AIza...", # required
  model: "gemini-2.0-flash", # default: "gemini-2.0-flash"
  system_prompt: "Be concise."
)
```

### DeepSeek

Supports chat and tool calling. File attachments are silently ignored (DeepSeek API does not support multimodal input).

```elixir
provider = ExAgent.Providers.DeepSeek.new(
  api_key: "sk-...",  # required
  model: "deepseek-chat", # default: "deepseek-chat"
  system_prompt: "You are a coding expert."
)
```

## Core Concepts

### Message

Represents a single message in a conversation.

```elixir
{:ok, msg} = ExAgent.Message.new(role: :user, content: "Hello!")
# Supported roles: :system, :user, :assistant, :tool
```

### Tool

Defines a function the LLM can invoke, with JSON Schema parameters.

```elixir
{:ok, tool} = ExAgent.Tool.new(
  name: "get_weather",
  description: "Get current weather for a city",
  parameters: %{
    "type" => "object",
    "properties" => %{
      "city" => %{"type" => "string", "description" => "City name"}
    },
    "required" => ["city"]
  },
  function: fn %{"city" => city} ->
    {:ok, "#{city}: 22C, sunny"}
  end
)
```

### Context

Portable conversation state with message history and metadata.

```elixir
context = ExAgent.Context.new(metadata: %{session_id: "abc123"})

{:ok, msg} = ExAgent.Message.new(role: :user, content: "Hello")
context = ExAgent.Context.add_message(context, msg)

# Get the last assistant response
last = ExAgent.Context.get_last_assistant_message(context)
```

### Skill

A loadable persona with its own system prompt, tools, and activation function.

```elixir
{:ok, sql_skill} = ExAgent.Skill.new(
  name: "sql_expert",
  system_prompt: "You are a SQL expert. Help users write queries.",
  tools: [sql_tool],
  activation_fn: fn ctx ->
    Enum.any?(ctx.messages, fn m ->
      String.contains?(m.content, "SQL") or String.contains?(m.content, "SELECT")
    end)
  end
)
```

## Agent Lifecycle

Agents are GenServer processes managed by a DynamicSupervisor.

```elixir
# Start an agent with tools and skills
{:ok, agent} = ExAgent.start_agent(
  provider: provider,
  id: "my-agent",
  tools: [weather_tool, search_tool],
  skills: [sql_skill]
)

# Synchronous chat
{:ok, response} = ExAgent.chat(agent, "What's the weather in Tokyo?")
IO.puts(response.content)

# Asynchronous chat
task = ExAgent.chat_async(agent, "Tell me a story")
{:ok, response} = Task.await(task)

# Inspect conversation history
context = ExAgent.get_context(agent)
Enum.each(context.messages, fn msg ->
  IO.puts("#{msg.role}: #{msg.content}")
end)

# Reset conversation
ExAgent.reset(agent)

# Stop the agent
ExAgent.stop_agent(agent)
```

## File Attachments

Send images, PDFs, and other files alongside chat messages. Files become part of the conversation context, so the LLM can reference them in follow-up messages. You can either send files inline (base64-encoded) or upload them first for better performance.

```elixir
# Attach a file by path (inline base64)
{:ok, response} = ExAgent.chat(agent, "Describe this image",
  files: [%{path: "photo.jpg", mime_type: "image/jpeg"}])

# Attach raw binary data (inline base64)
image_data = File.read!("diagram.png")
{:ok, response} = ExAgent.chat(agent, "What's in this diagram?",
  files: [%{data: image_data, mime_type: "image/png"}])

# Multiple files of any type
{:ok, response} = ExAgent.chat(agent, "Summarize these documents",
  files: [
    %{path: "report.pdf", mime_type: "application/pdf"},
    %{path: "data.csv", mime_type: "text/csv"},
    %{path: "notes.md", mime_type: "text/markdown"}
  ])

# Files persist in conversation context — the LLM remembers them
{:ok, _} = ExAgent.chat(agent, "Now focus on the second document")
```

### Supported File Types

| Type | MIME Type | OpenAI | Gemini | DeepSeek |
|------|-----------|--------|--------|----------|
| JPEG | `image/jpeg` | Yes | Yes | No |
| PNG | `image/png` | Yes | Yes | No |
| GIF | `image/gif` | Yes | Yes | No |
| WebP | `image/webp` | Yes | Yes | No |
| PDF | `application/pdf` | Yes | Yes | No |
| TXT | `text/plain` | Yes | Yes | No |
| Markdown | `text/markdown` | Yes | Yes | No |
| CSV | `text/csv` | Yes | Yes | No |

> **Note:** DeepSeek does not support multimodal input. File attachments on DeepSeek agents are silently ignored.

## File Uploads

For large files or when you want to reuse the same file across multiple conversations, upload the file first and reference it later. This avoids sending base64-encoded data with every chat request.

### Upload and Reference (OpenAI)

```elixir
provider = ExAgent.Providers.OpenAI.new(api_key: System.get_env("OPENAI_API_KEY"))

# Upload a file from disk
{:ok, ref} = ExAgent.upload_file(provider, "report.pdf", "application/pdf")

# Use the reference in chat — no base64 encoding, just a lightweight file ID
{:ok, agent} = ExAgent.start_agent(provider: provider)
{:ok, response} = ExAgent.chat(agent, "Summarize this report",
  files: [%{file_ref: ref}])

# Reuse the same reference in another message
{:ok, response} = ExAgent.chat(agent, "What are the key findings?",
  files: [%{file_ref: ref}])
```

### Upload and Reference (Gemini)

```elixir
provider = ExAgent.Providers.Gemini.new(api_key: System.get_env("GEMINI_API_KEY"))

# Upload a file — Gemini files expire after 48 hours
{:ok, ref} = ExAgent.upload_file(provider, "photo.jpg", "image/jpeg")

# Check if a reference has expired
ExAgent.FileRef.expired?(ref)

# Use in chat
{:ok, agent} = ExAgent.start_agent(provider: provider)
{:ok, response} = ExAgent.chat(agent, "Describe what you see",
  files: [%{file_ref: ref}])
```

### Upload Raw Binary Data

```elixir
# If you already have the file contents in memory
image_bytes = File.read!("screenshot.png")
{:ok, ref} = ExAgent.upload_data(provider, image_bytes, "image/png",
  filename: "screenshot.png")
```

### Mix Inline and Uploaded Files

```elixir
# You can combine both approaches in a single message
{:ok, ref} = ExAgent.upload_file(provider, "large_video.mp4", "video/mp4")
{:ok, response} = ExAgent.chat(agent, "Compare these",
  files: [
    %{file_ref: ref},                                          # uploaded reference
    %{path: "small_image.jpg", mime_type: "image/jpeg"}        # inline base64
  ])
```

## Built-in Provider Tools

Each LLM provider offers built-in tools that can be enabled via the `built_in_tools` option — either at agent creation (applies to all calls) or per-message (overrides agent default).

### Gemini

```elixir
# Google Search grounding — LLM can search the web for up-to-date info
{:ok, agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.Gemini.new(api_key: gemini_key),
  built_in_tools: [:google_search]
)
{:ok, response} = ExAgent.chat(agent, "What happened in tech news today?")

# Code execution — LLM can write and run Python code
{:ok, response} = ExAgent.chat(agent, "Calculate fibonacci(20)",
  built_in_tools: [:code_execution])

# URL context — LLM can fetch and analyze web pages
{:ok, response} = ExAgent.chat(agent, "Summarize this page",
  built_in_tools: [:url_context])

# Combine multiple built-in tools
{:ok, response} = ExAgent.chat(agent, "Research and compute",
  built_in_tools: [:google_search, :code_execution])
```

Available Gemini built-in tools: `:google_search`, `:code_execution`, `:url_context`

### OpenAI

```elixir
# Web search — LLM can search the web
{:ok, agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.OpenAI.new(api_key: openai_key),
  built_in_tools: [:web_search]
)
{:ok, response} = ExAgent.chat(agent, "What are the latest Elixir releases?")

# Web search with user location for localized results
{:ok, response} = ExAgent.chat(agent, "Best restaurants nearby",
  built_in_tools: [%{web_search: %{"city" => "San Francisco", "country" => "US"}}])
```

Available OpenAI built-in tools: `:web_search`

### DeepSeek

```elixir
# Thinking/reasoning mode — enables chain-of-thought reasoning
{:ok, agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.DeepSeek.new(
    api_key: deepseek_key,
    model: "deepseek-reasoner"
  ),
  built_in_tools: [:thinking]
)
{:ok, response} = ExAgent.chat(agent, "Solve this step by step: if x^2 + 3x - 10 = 0, what is x?")
```

Available DeepSeek built-in tools: `:thinking`

## Tool Calling

When you provide tools to an agent, the LLM can invoke them automatically. The agent runs a tool execution loop:

1. Sends messages + tool definitions to the LLM
2. If the LLM returns a `tool_call`, the agent executes the matching function
3. Appends the tool result as a `:tool` message
4. Calls the LLM again with the updated context
5. Repeats until the LLM returns a final text response (max 10 iterations)

```elixir
{:ok, search_tool} = ExAgent.Tool.new(
  name: "web_search",
  description: "Search the web for information",
  parameters: %{
    "type" => "object",
    "properties" => %{
      "query" => %{"type" => "string", "description" => "Search query"}
    },
    "required" => ["query"]
  },
  function: fn %{"query" => query} ->
    # Your search implementation here
    {:ok, "Results for: #{query}"}
  end
)

{:ok, calc_tool} = ExAgent.Tool.new(
  name: "calculator",
  description: "Evaluate a math expression",
  parameters: %{
    "type" => "object",
    "properties" => %{
      "expression" => %{"type" => "string"}
    },
    "required" => ["expression"]
  },
  function: fn %{"expression" => expr} ->
    {result, _} = Code.eval_string(expr)
    {:ok, to_string(result)}
  end
)

{:ok, agent} = ExAgent.start_agent(
  provider: provider,
  tools: [search_tool, calc_tool]
)

# The LLM can now decide to call these tools during conversation
{:ok, response} = ExAgent.chat(agent, "What is 42 * 37?")
```

## Multi-Agent Patterns

### 1. Subagents (Centralized Orchestration)

A main orchestrator agent delegates work to specialized subagents. Each subagent runs in isolation with a fresh context — no state leaks between calls.

```elixir
alias ExAgent.Patterns.Subagents

# Define specialized subagent specs
researcher = %{
  name: "researcher",
  description: "Research a topic and return findings",
  provider: ExAgent.Providers.Gemini.new(api_key: gemini_key),
  system_prompt: "You are a research specialist. Provide detailed findings.",
  tools: []
}

coder = %{
  name: "coder",
  description: "Write code based on specifications",
  provider: ExAgent.Providers.OpenAI.new(api_key: openai_key),
  system_prompt: "You are an expert programmer. Write clean, tested code.",
  tools: []
}

# Convert subagent specs into tools for the orchestrator
orchestrator_tools = Subagents.build_orchestrator_tools([researcher, coder])

# The orchestrator uses these as regular tools — when the LLM calls
# "researcher" or "coder", it spawns an ephemeral subagent call
{:ok, orchestrator} = ExAgent.start_agent(
  provider: ExAgent.Providers.OpenAI.new(
    api_key: openai_key,
    system_prompt: "You orchestrate tasks. Use the researcher for facts and the coder for code."
  ),
  tools: orchestrator_tools
)

{:ok, response} = ExAgent.chat(orchestrator, "Research Elixir GenServers and write an example")

# You can also invoke subagents directly
{:ok, result} = Subagents.invoke_subagent(researcher, "Explain OTP supervision trees")

# Or invoke multiple in parallel
results = Subagents.invoke_subagents_parallel([
  {researcher, "What is GenServer?"},
  {coder, "Write a GenServer example"}
])
# => [{"researcher", {:ok, "GenServer is..."}}, {"coder", {:ok, "defmodule..."}}]
```

### 2. Skills (Progressive Disclosure)

A single agent dynamically loads specialized system prompts and tools based on conversation context. Skills are evaluated before each LLM call.

```elixir
# Define skills with activation functions
{:ok, sql_skill} = ExAgent.Skill.new(
  name: "sql_expert",
  system_prompt: "You are a SQL expert. Help users write and optimize queries.",
  tools: [sql_execute_tool],
  activation_fn: fn ctx ->
    ctx.messages
    |> Enum.any?(fn m ->
      String.match?(m.content, ~r/SQL|SELECT|INSERT|UPDATE|DELETE|database/i)
    end)
  end
)

{:ok, python_skill} = ExAgent.Skill.new(
  name: "python_expert",
  system_prompt: "You are a Python expert. Write idiomatic Python code.",
  tools: [python_run_tool],
  activation_fn: fn ctx ->
    ctx.messages
    |> Enum.any?(fn m -> String.contains?(m.content, "Python") end)
  end
)

# Start agent with skills — it begins as a generalist
{:ok, agent} = ExAgent.start_agent(
  provider: provider,
  skills: [sql_skill, python_skill]
)

# When the user mentions SQL, the sql_expert skill activates automatically
{:ok, response} = ExAgent.chat(agent, "Help me write a SQL query to find active users")
# => Agent now uses the sql_expert system prompt and tools

# Skills can also be loaded dynamically at runtime
{:ok, new_skill} = ExAgent.Skill.new(name: "devops", system_prompt: "You are a DevOps expert.")
ExAgent.Agent.load_skill(agent, new_skill)
```

### 3. Handoffs (State-Driven Transitions)

The active agent changes dynamically. When the LLM invokes a handoff tool, control transfers to a different agent. The caller receives a `{:handoff, target, context}` tuple and decides where to route subsequent messages.

```elixir
alias ExAgent.Patterns.Handoff

# Start specialized agents
{:ok, sales_agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.OpenAI.new(
    api_key: key,
    system_prompt: "You are a sales specialist."
  )
)

{:ok, support_agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.OpenAI.new(
    api_key: key,
    system_prompt: "You are a technical support specialist."
  )
)

# Build handoff tools
handoff_to_support = Handoff.build_handoff_tool(
  "support",
  support_agent,
  "Transfer to technical support when the user has a technical issue"
)

handoff_to_sales = Handoff.build_handoff_tool(
  "sales",
  sales_agent,
  "Transfer to sales when the user wants to buy something"
)

# Start a triage agent with handoff tools
{:ok, triage_agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.OpenAI.new(
    api_key: key,
    system_prompt: "You are a triage agent. Route users to the right department."
  ),
  tools: [handoff_to_support, handoff_to_sales]
)

# When the LLM decides to hand off, you get a handoff tuple
case ExAgent.chat(triage_agent, "My app keeps crashing") do
  {:ok, response} ->
    # Normal response — agent handled it directly
    IO.puts(response.content)

  {:handoff, target_pid, context} ->
    # Transfer context and continue with the new agent
    ExAgent.handoff(target_pid, context)
    {:ok, response} = ExAgent.chat(target_pid, "My app keeps crashing")
    IO.puts(response.content)
end
```

### 4. Router (Parallel Dispatch & Synthesis)

Classifies input, dispatches to multiple specialized agents in parallel, and synthesizes results into a single response.

```elixir
alias ExAgent.Patterns.Router

# Start specialized agents
{:ok, code_agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.OpenAI.new(
    api_key: key,
    system_prompt: "Analyze code quality and suggest improvements."
  )
)

{:ok, security_agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.OpenAI.new(
    api_key: key,
    system_prompt: "Analyze code for security vulnerabilities."
  )
)

{:ok, perf_agent} = ExAgent.start_agent(
  provider: ExAgent.Providers.Gemini.new(
    api_key: gemini_key,
    system_prompt: "Analyze code for performance issues."
  )
)

# Define routes with match functions
routes = [
  %{name: "code_quality", agent: code_agent, match_fn: fn _ -> true end},
  %{name: "security", agent: security_agent, match_fn: &String.contains?(&1, "security")},
  %{name: "performance", agent: perf_agent, match_fn: &String.contains?(&1, "performance")}
]

# Route dispatches to all matching agents in parallel
{:ok, result} = ExAgent.route(
  "Review this code for security and performance issues: def fetch(url), do: HTTPoison.get!(url)",
  routes: routes,
  timeout: 30_000
)

IO.puts(result)
# ## code_quality
# The function lacks error handling...
#
# ## security
# Using get! will raise on HTTP errors...
#
# ## performance
# Consider connection pooling...

# Custom synthesizer
{:ok, result} = ExAgent.route("analyze this code",
  routes: routes,
  synthesizer: fn _input, results ->
    results
    |> Enum.map(fn {name, content} -> "**#{name}**: #{content}" end)
    |> Enum.join("\n\n")
  end
)
```

## Adding a Custom Provider

Any LLM can be integrated by defining a struct and implementing the `ExAgent.LlmProvider` protocol:

```elixir
defmodule MyApp.Providers.Anthropic do
  @moduledoc "Custom Anthropic Claude provider."

  defstruct [
    :api_key, :req,
    model: "claude-sonnet-4-20250514",
    base_url: "https://api.anthropic.com/v1",
    system_prompt: nil,
    tools: []
  ]

  def new(opts) do
    provider = struct!(__MODULE__, opts)
    %{provider | req: Req.new(
      base_url: provider.base_url,
      headers: [
        {"x-api-key", provider.api_key},
        {"anthropic-version", "2023-06-01"}
      ]
    )}
  end

  defimpl ExAgent.LlmProvider do
    def chat(provider, messages, _opts) do
      body = %{
        "model" => provider.model,
        "max_tokens" => 1024,
        "messages" => Enum.map(messages, fn msg ->
          %{"role" => to_string(msg.role), "content" => msg.content}
        end)
      }

      case Req.post(provider.req, url: "/messages", json: body) do
        {:ok, %Req.Response{status: 200, body: %{"content" => [%{"text" => text} | _]}}} ->
          {:ok, %ExAgent.Message{role: :assistant, content: text}}

        {:ok, %Req.Response{status: status, body: body}} ->
          {:error, {status, body}}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end
end

# Use it like any other provider
provider = MyApp.Providers.Anthropic.new(api_key: "sk-ant-...")
{:ok, agent} = ExAgent.start_agent(provider: provider)
{:ok, response} = ExAgent.chat(agent, "Hello Claude!")
```

## Architecture

### Supervision Tree

```
Application (ex_agent)
  |
  ExAgent.AgentSupervisor (:one_for_one)
    |
    +-- ExAgent.AgentDynamicSupervisor (:one_for_one)
    |     |
    |     +-- ExAgent.Agent (id: "orchestrator")
    |     +-- ExAgent.Agent (id: "coder")
    |     +-- ExAgent.Agent (id: "reviewer")
    |     +-- ... (any runtime agents)
    |
    +-- ExAgent.TaskSupervisor
          |
          +-- Task (async chat calls)
          +-- Task (parallel subagent invocations)
          +-- Task (router parallel dispatch)
```

### Design Decisions

- **Protocol dispatch** — Provider structs implement `ExAgent.LlmProvider`, enabling compile-time polymorphism
- **Thin protocol impls** — Protocol implementations delegate to service modules under `services/`, keeping HTTP logic separate
- **Tool loop in GenServer** — The `handle_call({:chat, ...})` contains the tool execution loop, processing one turn at a time to prevent race conditions on context
- **Subagents bypass GenServer** — Ephemeral stateless calls use `LlmProvider.chat/3` directly in supervised Tasks
- **Handoff returns to caller** — Keeps agents decoupled; the caller decides routing after a handoff
- **Router is a plain module** — Stateless classify-dispatch-synthesize flow needs no GenServer
- **All patterns share one Agent GenServer** — Patterns augment behavior through state and tools, not separate process types

### Project Structure

```
lib/
  ex_agent.ex                     # Public API facade
  ex_agent/
    llm_provider.ex               # LlmProvider protocol
    file_uploader.ex              # FileUploader protocol
    file_ref.ex                   # %FileRef{} struct (uploaded file reference)
    message.ex                    # %Message{} struct
    tool.ex                       # %Tool{} struct
    context.ex                    # %Context{} struct
    skill.ex                      # %Skill{} struct
    agent.ex                      # Agent GenServer
    supervisor.ex                 # AgentSupervisor
    dynamic_supervisor.ex         # AgentDynamicSupervisor
    providers/
      openai.ex                   # OpenAI provider + LlmProvider + FileUploader
      gemini.ex                   # Gemini provider + LlmProvider + FileUploader
      deep_seek.ex                # DeepSeek provider + LlmProvider
    services/
      openai_service.ex           # OpenAI chat HTTP calls via Req
      openai_upload_service.ex    # OpenAI file upload (POST /v1/files)
      gemini_service.ex           # Gemini chat HTTP calls via Req
      gemini_upload_service.ex    # Gemini file upload (Files API)
      deep_seek_service.ex        # DeepSeek HTTP calls via Req
    patterns/
      subagents.ex                # Centralized orchestration
      skills.ex                   # Progressive disclosure
      handoff.ex                  # State-driven transitions
      router.ex                   # Parallel dispatch & synthesis
```

## License

MIT
