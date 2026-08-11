# ExAgent

Build multi-agent LLM applications in Elixir. One behaviour abstracts OpenAI, Gemini, and
any OpenAI-compatible endpoint; OTP primitives orchestrate them.

[Hex](https://hex.pm/packages/ex_agent) · [HexDocs](https://hexdocs.pm/ex_agent) · [![CI](https://github.com/tiagodavi/ex_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/tiagodavi/ex_agent/actions/workflows/ci.yml)

- **Swap providers without touching application code** — one behaviour, struct-based config
- **Config-driven roles** — name a purpose (`:vision`, `:embed`), not a vendor
- **Structured streaming** — text, reasoning, tool-call deltas, usage as typed chunks
- **Automatic tool execution** — define a tool once; the agent loops until done
- **Multimodal** — images, PDFs, video, audio from a path, bytes, a URL, or an upload
- **Embeddings for RAG** — one task vocabulary across providers, provenance on every result
- **Normalized errors** — one `%ExAgent.Error{}` with a `retryable?` flag
- **Fails loudly** — unsupported input is rejected before a request is built, never dropped
- **4 multi-agent patterns** — Subagents, Skills, Handoffs, Router

## Install

```elixir
def deps do
  [{:ex_agent, "~> 0.3.0"}]
end
```

## Quick start

```elixir
provider = ExAgent.Providers.OpenAI.new(api_key: System.fetch_env!("OPENAI_API_KEY"))
{:ok, agent} = ExAgent.start_agent(provider: provider)

{:ok, response} = ExAgent.chat(agent, "What is Elixir?")
response.content   #=> "Elixir is a functional programming language..."
```

---

## Contents

| | |
|---|---|
| [Providers](#providers) · [Roles](#roles) | Configure who answers |
| [Agents](#agents) · [Tools](#tools) · [Streaming](#streaming) | Run a conversation |
| [Attachments](#attachments) · [Uploads](#uploads) | Send files |
| [Embeddings](#embeddings) | RAG |
| [Patterns](#multi-agent-patterns) | Subagents, Skills, Handoffs, Router |
| [Observability](#observability) | Telemetry events |
| [Recipes](#recipes) | Retry, RAG, multi-tenant, LiveView, testing |
| [Custom providers](#custom-providers) · [Architecture](#architecture) · [Testing](#testing) | Extend |

---

## Providers

Every provider is a struct built by `new/1`, which validates options and prepares a `Req`
client. Pass it to `start_agent/1` or use it directly.

```elixir
ExAgent.Providers.OpenAI.new(
  api_key: "sk-...",                  # required
  model: "gpt-4o",                    # default
  system_prompt: "You are concise.",
  temperature: 0.6,               # omitted from the request when unset
  max_tokens: 4096                # omitted when unset, so the model's own ceiling applies
)

ExAgent.Providers.Gemini.new(
  api_key: "AIza...",                 # required
  model: "gemini-3.6-flash"           # default
)
```

### OpenAI-compatible (vLLM, Modal, OpenRouter, Together, Groq)

One provider for any endpoint speaking the OpenAI chat-completions dialect. Unlike
`Providers.OpenAI`, it accepts arbitrary auth headers — which is what makes Modal's proxy
auth reachable.

```elixir
provider = ExAgent.Providers.OpenAICompatible.new(
  base_url: System.fetch_env!("MODAL_URL") <> "/v1",   # required, include /v1
  model: "Qwen/Qwen3-VL-8B-Instruct",                  # required
  headers: [
    {"Modal-Key", System.fetch_env!("MODAL_KEY")},
    {"Modal-Secret", System.fetch_env!("MODAL_SECRET")}
  ],
  modalities: [:text, :image, :video],
  max_inline_bytes: 33_554_432
)

# Gateways using bearer auth: :api_key is sugar for Authorization: Bearer.
# Explicit :headers override it rather than being sent alongside.
ExAgent.Providers.OpenAICompatible.new(
  base_url: "https://openrouter.ai/api/v1",
  model: "meta-llama/llama-3.3-70b-instruct",
  api_key: System.fetch_env!("OPENROUTER_API_KEY"),
  modalities: [:text, :image]
)

# Container swaps where config still names the old model are a common failure.
case ExAgent.Providers.OpenAICompatible.probe(provider) do
  :ok -> :ready
  {:error, error} -> Logger.warning(Exception.message(error))
end
```

There is **no Files API** here: bytes always become `data:` URIs, and anything past
`:max_inline_bytes` returns `{:error, %ExAgent.Error{type: :unsupported}}` telling you to
host it at a URL the container can reach — never truncated, never retried. A
`%{file_ref: ref}` from another provider is rejected for the same reason.

### Modalities are per model, not per vendor

`o1-mini` reads no images; a text-only Gemini could ship tomorrow. Every provider takes
`:modalities`, and narrowing makes the gate fire before any request is built.

```elixir
provider = ExAgent.Providers.OpenAI.new(api_key: key, model: "o1-mini", modalities: [:text])
{:ok, agent} = ExAgent.start_agent(provider: provider)

ExAgent.chat(agent, "Describe this", files: [%{path: "photo.jpg"}])
#=> {:error, %ExAgent.Error{type: :unsupported}}
```

| Modality | OpenAI | Gemini | OpenAICompatible |
|---|---|---|---|
| `:text` | ✅ | ✅ | ✅ |
| `:image` | ✅ | ✅ | declare it |
| `:document` | ✅ | ✅ | declare it |
| `:video` | ❌ | ✅ | declare it |
| `:audio` | ❌ | ✅ | declare it |

`:text` is always present — it is what a plain message is. The rest are *defaults* you can
narrow or widen. ExAgent keeps no model-to-modality table on purpose: it would go stale
silently, and whoever picked the model already knows.

"Declare it" means `OpenAICompatible` ships `[:text]` and you list what your deployment
actually serves — including `:document`, for a gateway fronting a document-reading model:

```elixir
ExAgent.Providers.OpenAICompatible.new(
  base_url: "https://openrouter.ai/api/v1",
  model: "anthropic/claude-sonnet-4.5",
  api_key: key,
  modalities: [:text, :image, :document]
)
```

Nothing here is validated against the endpoint until a request is made, so declare only
what the served model actually reads.

## Roles

Declare which provider serves which purpose in config, then name the purpose at the call
site. Swapping a hosted model for a self-hosted container becomes a one-file change.

```elixir
# config/runtime.exs
config :ex_agent, :roles,
  chat: {ExAgent.Providers.Gemini, api_key: System.fetch_env!("GEMINI_API_KEY")},
  vision: {ExAgent.Providers.OpenAICompatible,
             base_url: System.fetch_env!("MODAL_URL") <> "/v1",
             model: "Qwen/Qwen3-VL-8B-Instruct",
             modalities: [:text, :image]},
  embed: {ExAgent.Providers.OpenAICompatible,
            base_url: System.fetch_env!("JINA_URL") <> "/v1",
            model: "jinaai/jina-embeddings-v3"}
```

Role names are arbitrary atoms — `:ocr`, `:cheap_summarizer`, whatever fits. A bare module
means "that module with no options".

```elixir
# Roles are sugar: provider!/1 returns an ordinary struct usable everywhere.
{:ok, agent} = ExAgent.start_agent(provider: ExAgent.provider!(:vision))
{:ok, agent} = ExAgent.start_agent(role: :vision, tools: [...])   # shorthand

ExAgent.roles()          #=> [:chat, :vision, :embed]
ExAgent.provider(:nope)  #=> :error   (non-raising counterpart)
```

Stateless one-shot wrappers skip the agent process entirely:

```elixir
{:ok, response} = ExAgent.chat_with(:vision, "Invoice total?",
                    files: [%{url: "https://cdn.example.com/invoice.png"}])

ExAgent.stream_with(:chat, "Explain OTP") |> Enum.each(&IO.write(&1.text || ""))

{:ok, docs} = ExAgent.embed_with(:embed, chunks, task: :retrieval_document)
```

**These do not run the tool loop** — it lives in the agent. A tool-configured provider
returns the raw `{:tool_calls, calls}`. Use `start_agent(role: ..., tools: [...])`
when you want tools resolved.

Roles resolve **once at boot** into `:persistent_term`, so lookups cost nothing per
request. A module that is missing, lacks `new/1`, does not implement the behaviour, or
whose `new/1` raises fails the boot naming the role — a missing `GEMINI_API_KEY` crashes
at deploy time, not with a 401 at 3am. Vault-backed credentials can be a zero-arity
function or `{m, f, a}`, resolved once at boot:

```elixir
config :ex_agent, :roles,
  chat: {ExAgent.Providers.Gemini, api_key: {MyApp.Vault, :fetch, ["gemini"]}}
```

`provider!/2` merges overrides and returns a *fresh* struct — for per-tenant keys. It
rebuilds the `Req` client, so it is fine per request but not in a tight loop:

```elixir
ExAgent.provider!(:chat, api_key: tenant.openai_key, model: "gpt-4o-mini")
```

If you assemble role config after boot, call `ExAgent.Roles.build!/0` yourself.

## Agents

Agents are GenServers under a DynamicSupervisor. They hold conversation context and run
the tool loop.

```elixir
{:ok, agent} = ExAgent.start_agent(
  provider: provider,
  id: "support-42",
  tools: [weather_tool],
  skills: [sql_skill],
  built_in_tools: [:web_search],
  name: {:global, "support-42"}       # optional GenServer registration
)

{:ok, response} = ExAgent.chat(agent, "What's the weather in Tokyo?")
task = ExAgent.chat_async(agent, "Tell me a story");  {:ok, response} = Task.await(task)

ExAgent.get_context(agent).messages   # full history
ExAgent.reset(agent)                  # clear context
ExAgent.stop_agent(agent)
```

One turn at a time: while a request is in flight, `chat/3` returns `{:error, :busy}`.

### Multi-turn conversations

An agent **is** the conversation. Every turn is appended to its context and the whole
history is resent on the next request — you never assemble a message list yourself.

```elixir
{:ok, agent} = ExAgent.start_agent(provider: provider)

{:ok, _} = ExAgent.chat(agent, "My name is Ada.")
{:ok, r} = ExAgent.chat(agent, "What is my name?")
r.content   #=> "Your name is Ada."

Enum.map(ExAgent.get_context(agent).messages, & &1.role)
#=> [:user, :assistant, :user, :assistant]

ExAgent.reset(agent)   # forget everything; the system prompt still applies
```

Attachments are part of that history too, so a follow-up turn can refer back to a file
sent earlier without resending it.

History is **unbounded by default** — every turn resends the whole transcript, so cost
climbs turn over turn until the model returns `:context_length`. Cap it when a
conversation is long-lived:

```elixir
{:ok, agent} = ExAgent.start_agent(provider: provider, max_history: 20)
```

Leading system messages always survive the window, and a tool result is never left
without the assistant message that requested it. Trimming is opt-in because silently
forgetting what a user said is your decision, not the library's.

### System prompts and instructions

The system prompt lives on the **provider struct**, not in the context. It is prepended to
every request and is never stored as a message, so `reset/1` cannot lose it and it cannot
be pushed out of the window by history.

```elixir
provider = ExAgent.Providers.OpenAI.new(
  api_key: key,
  system_prompt: "You are a terse assistant. Answer in one sentence."
)

{:ok, agent} = ExAgent.start_agent(provider: provider)
```

Three ways to vary instructions, in increasing order of dynamism:

```elixir
# 1. Fixed for this agent — set it on the provider, as above.

# 2. Per conversation — build a provider per agent
{:ok, pirate} = ExAgent.start_agent(provider: %{provider | system_prompt: "Talk like a pirate."})

# 3. Conditional on the conversation — a Skill swaps in a prompt (and tools)
#    when its activation_fn matches. See "Skills" below.
{:ok, agent} = ExAgent.start_agent(provider: provider, skills: [sql_skill])
```

To steer a single turn without changing the agent, put the instruction in the message —
it becomes part of the history like any other user turn:

```elixir
ExAgent.chat(agent, "Reply as JSON only.\n\nList three Elixir books.")
```

### Seeding or restoring history

Rehydrate a conversation from your database — build a `Context` and hand it to a fresh
agent. This is the same mechanism the Handoff pattern uses.

```elixir
context =
  Enum.reduce(rows_from_db, ExAgent.Context.new(), fn row, ctx ->
    {:ok, msg} = ExAgent.Message.new(role: row.role, content: row.content)
    ExAgent.Context.add_message(ctx, msg)
  end)

{:ok, agent} = ExAgent.start_agent(provider: provider)
:ok = ExAgent.handoff(agent, context)          # replaces the agent's context

{:ok, r} = ExAgent.chat(agent, "What did we decide?")
```

Persisting works the other way round — `get_context/1` after each turn:

```elixir
for %ExAgent.Message{role: role, content: content} <- ExAgent.get_context(agent).messages do
  Repo.insert!(%Turn{conversation_id: id, role: to_string(role), content: content})
end
```

`ExAgent.Context.get_last_assistant_message/1` pulls just the latest reply, and
`ExAgent.Context.new(metadata: %{...})` carries your own bookkeeping alongside the
messages.

### Response

`chat/3` and `collect/1` both return the same struct, so streaming and non-streaming share
one downstream path.

```elixir
response.content        #=> "Elixir is a functional language..."
response.usage          #=> %{input_tokens: 8, output_tokens: 96, total_tokens: 104}
response.finish_reason  #=> :stop | :length | :tool_calls | :content_filter | :error
response.tool_calls     #=> nil, or [%{"id" => ..., "name" => ..., "args" => %{}}]
response.thinking       #=> reasoning trace, when the model emitted one
response.message        #=> the %ExAgent.Message{} appended to history
```

`:thinking` is deliberately kept out of `:message` — reasoning is not conversation
history, and replaying it corrupts the next turn.

### Errors

Every failed operation returns `{:error, %ExAgent.Error{}}` in one vocabulary, so retry
logic is written once rather than per provider.

```elixir
case ExAgent.chat(agent, "Summarize this") do
  {:ok, response} -> response
  {:error, %ExAgent.Error{retryable?: true} = e} -> retry_with_backoff(e)
  {:error, %ExAgent.Error{type: :context_length}} -> summarize_history_and_retry()
  {:error, %ExAgent.Error{} = e} -> Logger.error(Exception.message(e))
end
```

| Type | Meaning | `retryable?` |
|---|---|---|
| `:auth` | Bad or missing credentials (401, 403) | ❌ |
| `:not_found` | Unknown model or resource (404) | ❌ |
| `:timeout` | Request timed out (408, transport) | ✅ |
| `:rate_limit` | Rate or quota limit hit (429) | ✅ |
| `:context_length` | Input exceeds the context window | ❌ |
| `:invalid_request` | Malformed request (other 4xx) | ❌ |
| `:unsupported` | The provider cannot do this at all | ❌ |
| `:server` | Provider-side failure (5xx, bad shape) | ✅ |
| `:transport` | Connection-level failure | ✅ |

`:raw` carries the provider's original body and `:provider` names the module. `Error` is
also an exception, so it can be raised where no return value exists.

## Tools

Define a function the LLM can invoke. The agent loops — call, execute, feed the result
back — until a final answer or 10 iterations (`max_tool_iterations:` to change it).

```elixir
{:ok, weather_tool} = ExAgent.Tool.new(
  name: "get_weather",
  description: "Get current weather for a city",
  parameters: %{
    "type" => "object",
    "properties" => %{"city" => %{"type" => "string"}},
    "required" => ["city"]
  },
  function: fn %{"city" => city} -> {:ok, "#{city}: 22C, sunny"} end
)

{:ok, agent} = ExAgent.start_agent(provider: provider, tools: [weather_tool])
{:ok, response} = ExAgent.chat(agent, "Should I take a coat in Tokyo?")
```

A tool returns `{:ok, result}`, `{:error, reason}` (fed back to the LLM as an error), or
any bare value — which is taken as the result. A result that is not a string is JSON
encoded, so returning a map or list from an API-backed tool is fine.

A model can request **several tools in one turn**. All of them run, in order, each with
its own result correlated back by the provider's call id.

### Built-in provider tools

Enable at agent creation (all calls) or per message (overrides the agent default).

```elixir
# Gemini: :google_search, :code_execution, :url_context
{:ok, agent} = ExAgent.start_agent(provider: gemini, built_in_tools: [:google_search])
{:ok, r} = ExAgent.chat(agent, "Compute fibonacci(20)", built_in_tools: [:code_execution])

# OpenAI: :web_search. Two constraints come from OpenAI, not ExAgent —
# only a *-search-preview model accepts it, and those models reject `temperature`.
provider = ExAgent.Providers.OpenAI.new(
  api_key: key, model: "gpt-4o-search-preview", temperature: nil
)
{:ok, agent} = ExAgent.start_agent(provider: provider, built_in_tools: [:web_search])

# ...with location for localized results
ExAgent.chat(agent, "Best restaurants nearby",
  built_in_tools: [%{web_search: %{"city" => "Lisbon", "country" => "PT"}}])
```

ExAgent forwards `temperature` as configured rather than dropping it when a model objects
— a silently ignored sampling parameter is worse than a 400 naming the field.

## Streaming

```elixir
agent
|> ExAgent.chat_stream("Explain OTP supervision step by step")
|> Enum.each(fn
  %ExAgent.Chunk{type: :text_delta, text: text} -> IO.write(text)
  %ExAgent.Chunk{type: :thinking_delta, text: t} -> IO.write(IO.ANSI.faint() <> t)
  %ExAgent.Chunk{type: :done, finish_reason: reason} -> IO.puts("\n[#{reason}]")
  _chunk -> :ok
end)
```

| Chunk type | Carries |
|---|---|
| `:text_delta` | `:text` — a piece of the answer |
| `:thinking_delta` | `:text` — a piece of the reasoning trace |
| `:tool_call_delta` | `:index`, `:id`, `:name`, `:arguments` |
| `:usage` | `:usage` — token counts |
| `:done` | `:finish_reason`, plus `:error` when the stream failed |

**Streaming never raises.** A busy agent, an HTTP error, a dropped connection, and an idle
timeout all arrive as the terminal `:done` chunk carrying an `ExAgent.Error`. Every stream
ends with exactly one `:done`, and whatever text already arrived stays valid.

```elixir
agent
|> ExAgent.chat_stream("Summarize this")
|> Enum.each(fn
  %ExAgent.Chunk{type: :text_delta, text: t} -> IO.write(t)
  %ExAgent.Chunk{type: :done, error: %ExAgent.Error{retryable?: true}} -> retry()
  %ExAgent.Chunk{type: :done, error: %ExAgent.Error{} = e} -> Logger.error(e.message)
  _ -> :ok
end)
```

`:arguments` is a **raw JSON fragment**, not a decoded map — providers split function
arguments across chunks, concatenated by `:index`. `collect/1` handles that:

```elixir
{:ok, response} = agent |> ExAgent.chat_stream("Explain OTP") |> ExAgent.collect()
```

Tool-call turns resolve first without streaming; only the final turn streams. Context is
committed when the stream is **fully consumed**, so the stream must be consumed.

Streaming straight from a provider skips the agent, its context, and the tool loop:

```elixir
{:ok, msg} = ExAgent.Message.new(role: :user, content: "Why is the sky blue?")
provider |> ExAgent.Provider.stream([msg]) |> Enum.each(&IO.write(&1.text || ""))
```

## Attachments

An attachment carries exactly one source — `:path`, `:data`, `:url`, or `:file_ref`. Files
stay in conversation context, so follow-up turns can reference them.

```elixir
ExAgent.chat(agent, "Describe this", files: [%{path: "photo.jpg"}])
ExAgent.chat(agent, "What's here?",  files: [%{data: File.read!("d.png")}])
ExAgent.chat(agent, "Read this",     files: [%{url: "https://cdn.example.com/inv.png"}])
ExAgent.chat(agent, "Summarize",     files: [%{file_ref: ref}])

# Mix sources and types freely
ExAgent.chat(agent, "Compare these", files: [
  %{path: "report.pdf"},
  %{url: "https://cdn.example.com/data.csv"},
  %{data: notes, mime_type: "text/markdown"}
])

# The LLM still remembers them next turn
ExAgent.chat(agent, "Now focus on the second document")
```

### MIME types

Inferred from the extension for `:path` and `:url` (query strings ignored) and from magic
bytes for `:data`. **ExAgent never guesses** — when inference fails you get
`{:error, %ExAgent.Error{type: :invalid_request}}` naming `:mime_type`.

```elixir
# inferred: image/png (the query string is ignored)
ExAgent.chat(agent, "Read this", files: [%{url: "https://cdn.example.com/s.png?sig=abc"}])

# explicit: the extension says nothing useful
ExAgent.chat(agent, "Parse this", files: [%{path: "/tmp/export", mime_type: "text/csv"}])

# explicit: a CDN URL with no extension at all
ExAgent.chat(agent, "Describe this",
  files: [%{url: "https://picsum.photos/id/237/800/600", mime_type: "image/jpeg"}])
```

### Inline or uploaded — chosen for you

| Source | Delivery |
|---|---|
| `:url` | Referenced as-is — **never fetched** by ExAgent |
| `:file_ref` | Referenced as-is — already uploaded |
| Bytes under the inline ceiling | Base64 `data:` URI |
| Bytes over it | Uploaded through the Files API, then referenced |

**Gemini** inlines to 20 MB (50 MB for PDFs), then uses the Files API and waits for
`ACTIVE` — large files and video sit in `PROCESSING`, and referencing them early fails.
**OpenAI** inlines to 20 MB, then uploads documents and references them by `file_id`.

> #### OpenAI cannot reference an uploaded image {: .warning}
>
> Chat completions has no content part for one: `image_file` is the Assistants API's
> shape, `image_url` requires a real URL, and the `file` part accepts PDFs only. An image
> over the inline ceiling, or an image `file_ref`, therefore returns
> `{:error, %ExAgent.Error{type: :unsupported}}` rather than spending an upload on a
> request that would always fail. Resize it, or host it and pass `%{url: ...}`.

Uploads are cached by content hash, scoped per provider + base URL + API key, so the same
file across turns uploads once and two accounts never share a reference. An expired
`FileRef` is re-uploaded automatically.

```elixir
ExAgent.Providers.Gemini.new(api_key: key, upload_cache: false)   # opt out
```

`:path` reads only the file size up front; bytes are read at request time, so a 40 MB
video is not carried in history every turn.

### Video

```elixir
ExAgent.chat(agent, "Summarize both clips",
  files: [
    %{url: "https://cdn.example.com/clip.mp4", fps: 2},
    %{path: "talk.mp4", provider_opts: %{"start_offset" => "10s"}}
  ])
```

`:fps` maps to Gemini's `video_metadata` and rides along on OpenAI-compatible video parts.
Anything string-keyed under `:provider_opts` is merged into the media part verbatim.
`:max_frames` is normalized but has no field in Gemini's API, so Gemini ignores it.

### Supported types

| Type | MIME | Modality |
|---|---|---|
| JPEG / PNG / GIF / WebP | `image/*` | `:image` |
| PDF / TXT / Markdown / CSV | `application/pdf`, `text/*` | `:document` |
| MP4 / QuickTime / WebM | `video/*` | `:video` |
| MP3 / WAV / M4A / FLAC | `audio/*` | `:audio` |

Custom providers declare support via `c:ExAgent.Provider.supported_modalities/1`. Omitting
it means text only — a provider that has not opted in fails loudly rather than quietly
discarding attachments.

## Uploads

Upload once, reference many times — no base64 on every request.

```elixir
{:ok, ref} = ExAgent.upload_file(provider, "report.pdf", "application/pdf")
{:ok, ref} = ExAgent.upload_data(provider, File.read!("shot.png"), "image/png",
                                 filename: "shot.png")

ExAgent.chat(agent, "Summarize this report", files: [%{file_ref: ref}])
ExAgent.chat(agent, "What are the key findings?", files: [%{file_ref: ref}])

ExAgent.FileRef.expired?(ref)   # Gemini files expire after 48h

# Mix uploaded and inline in one message
ExAgent.chat(agent, "Compare these", files: [
  %{file_ref: video_ref},
  %{path: "thumbnail.jpg"}
])
```

## Embeddings

Stateless — takes a provider struct, no agent.

```elixir
provider = ExAgent.Providers.Gemini.new(api_key: System.fetch_env!("GEMINI_API_KEY"))

{:ok, docs}  = ExAgent.embed(provider, ["Elixir is functional", "OTP supervises"],
                             task: :retrieval_document)
{:ok, query} = ExAgent.embed(provider, "what supervises processes?",
                             task: :retrieval_query)

docs.vectors
|> Enum.map(&ExAgent.Embeddings.cosine_similarity(&1, hd(query.vectors)))
|> Enum.with_index()
|> Enum.max_by(&elem(&1, 0))
```

### Store provenance with every vector

> **Embedding spaces are model-scoped.** Vectors from different models, dimensions, or
> tasks are not comparable, and mixing them degrades retrieval *silently* rather than
> failing. The only fix is a full re-embed.

```elixir
{:ok, result} = ExAgent.embed(provider, chunks, task: :retrieval_document)

Enum.zip(chunks, result.vectors)
|> Enum.map(fn {chunk, vector} ->
  %{text: chunk, embedding: vector,
    model: result.model, dimensions: result.dimensions, task: result.task}
end)
```

That turns a later model change into a detectable migration instead of a quiet regression.

### Tasks

`:task` says what the embedding is *for*. Query and document are deliberately distinct —
asymmetric retrieval needs both, and the wrong one degrades recall invisibly.

`:retrieval_query` · `:retrieval_document` · `:similarity` · `:classification` ·
`:clustering` · `:question_answering` · `:fact_verification` · `:code_query`

Providers express this incompatibly and ExAgent translates: `gemini-embedding-001` takes a
`taskType` enum; `gemini-embedding-2` has no such field and writes a text prefix (with
`:title` filling the document template); **OpenAI has no task support and errors** rather
than dropping it; OpenAI-compatible endpoints take a `task` body field.

```elixir
ExAgent.embed(provider, [%{content: "body", title: "OTP Guide"}],
  model: "gemini-embedding-2", task: :retrieval_document)
```

#### When the atom vocabulary does not fit

A self-hosted endpoint serves whatever model you deployed, and those vocabularies change
between versions — Jina v3 has `"retrieval.passage"`, while v5 replaced it with a
`"retrieval"` task plus a query/document prompt and added `"text-matching"`. So
`OpenAICompatible` — and only it — accepts a raw string:

```elixir
# Keep the portable atom, retarget the strings for your model
ExAgent.embed(jina, ["query"],
  task: :retrieval_query,
  task_map: %{retrieval_query: "retrieval.query", retrieval_document: "retrieval.passage"})

# Or pass the model's own task — sent verbatim, no translation, no validation
ExAgent.embed(jina, ["a"], task: "text-matching")
```

**Gemini and OpenAI take atoms only.** Gemini's `taskType` is a closed enum and OpenAI has
no task field at all, so there a string is a typo far more often than a new value and is
rejected naming the valid atoms. Model drift there is handled by `:embedding_family`
instead.

An atom is always validated, so `:retreival_query` is rejected rather than quietly sent.
Either way the result's `:task` carries back exactly what you passed, so stored provenance
stays truthful.

### Models and dimensions

`:model` is always the *embedding* model, never the provider's chat model — defaults are
`text-embedding-3-small` (OpenAI) and `gemini-embedding-001` (Gemini).

`:dimensions` truncates where supported. `gemini-embedding-001` does not renormalize a
truncated vector, so ExAgent rescales client-side; cosine similarity assumes unit length.

An unrecognized Gemini embedding model **errors rather than guessing** — sending a
`taskType` to a model that ignores it returns HTTP 200 with plausible floats that land in
your index and retrieve worse forever. Use `:embedding_family` (`:task_type` or `:prefix`)
to adopt a model this version predates.

## Multi-agent patterns

### Subagents — centralized orchestration

Each subagent runs in isolation with a fresh context; no state leaks between calls.

```elixir
alias ExAgent.Patterns.Subagents

researcher = %{name: "researcher", description: "Research a topic",
               provider: gemini, system_prompt: "Provide detailed findings."}
coder = %{name: "coder", description: "Write code",
          provider: openai, system_prompt: "Write clean, tested code."}

# As orchestrator tools — the LLM decides when to delegate
{:ok, orchestrator} = ExAgent.start_agent(
  provider: openai,
  tools: Subagents.build_orchestrator_tools([researcher, coder])
)
{:ok, r} = ExAgent.chat(orchestrator, "Research GenServers and write an example")

# Or drive them yourself
{:ok, result} = Subagents.invoke_subagent(researcher, "Explain supervision trees")

results = Subagents.invoke_subagents_parallel([
  {researcher, "What is GenServer?"},
  {coder, "Write a GenServer example"}
])
#=> [{"researcher", {:ok, "..."}}, {"coder", {:ok, "..."}}]
```

### Skills — progressive disclosure

One agent loads specialized prompts and tools based on context, evaluated before each call.

```elixir
{:ok, sql_skill} = ExAgent.Skill.new(
  name: "sql_expert",
  system_prompt: "You are a SQL expert.",
  tools: [sql_execute_tool],
  activation_fn: fn ctx ->
    Enum.any?(ctx.messages, &String.match?(&1.content, ~r/SQL|SELECT|database/i))
  end
)

{:ok, agent} = ExAgent.start_agent(provider: provider, skills: [sql_skill])
{:ok, r} = ExAgent.chat(agent, "Write a query to find active users")  # skill activates

ExAgent.Agent.load_skill(agent, another_skill)   # add one at runtime
```

Skills are re-evaluated before **every** turn, so one that stops matching is undone and
the agent's own `system_prompt` comes back. A skill activating once does not repaint the
agent for the rest of its life.

### Handoffs — state-driven transitions

The caller receives `{:handoff, target, context}` and decides routing, so agents stay
decoupled.

```elixir
alias ExAgent.Patterns.Handoff

to_support = Handoff.build_handoff_tool("support", support_agent,
               "Transfer when the user has a technical issue")

{:ok, triage} = ExAgent.start_agent(provider: provider, tools: [to_support])

case ExAgent.chat(triage, "My app keeps crashing") do
  {:ok, response} ->
    response.content

  {:handoff, target, context} ->
    ExAgent.handoff(target, context)
    {:ok, response} = ExAgent.chat(target, "My app keeps crashing")
    response.content
end
```

### Router — parallel dispatch and synthesis

Routes are matched by `match_fn` (not an LLM classifier), dispatched in parallel, then
synthesized.

```elixir
routes = [
  %{name: "quality",  agent: code_agent,     match_fn: fn _ -> true end},
  %{name: "security", agent: security_agent, match_fn: &String.contains?(&1, "security")},
  %{name: "perf",     agent: perf_agent,     match_fn: &String.contains?(&1, "performance")}
]

{:ok, result} = ExAgent.route("Review this for security issues: ...",
                  routes: routes, timeout: 30_000)

# Default synthesizer joins with "## name" headers; override it
{:ok, result} = ExAgent.route("analyze this",
  routes: routes,
  synthesizer: fn _input, results ->
    Enum.map_join(results, "\n\n", fn {name, content} -> "**#{name}**: #{content}" end)
  end
)

ExAgent.route("anything", routes: [%{name: "n", agent: a, match_fn: fn _ -> false end}])
#=> {:error, :no_matching_routes}
```

---

## Observability

Every provider call emits `:telemetry` events, so latency and token spend are
measurable without wrapping the library. Nothing is logged on your behalf.

```elixir
:telemetry.attach_many(
  "ex-agent",
  [[:ex_agent, :chat, :stop], [:ex_agent, :embed, :stop], [:ex_agent, :tool, :stop]],
  &MyApp.Metrics.handle/4,
  nil
)

def handle([:ex_agent, :chat, :stop], measurements, metadata, _config) do
  MyApp.Metrics.histogram("llm.latency", measurements.duration, tags: [metadata.model])
  MyApp.Metrics.count("llm.tokens", measurements[:total_tokens] || 0)
end
```

`:result` is `:ok`, `:tool_calls`, or `:error`; a failure also carries `:error_type` and
`:retryable?`, which is what a retry-rate dashboard is built on. See `ExAgent.Telemetry`
for the full event table.

## Recipes

### Retry with backoff on transient failures

```elixir
def chat_with_retry(agent, input, attempt \\ 1) do
  case ExAgent.chat(agent, input) do
    {:error, %ExAgent.Error{retryable?: true}} when attempt < 4 ->
      Process.sleep(trunc(:math.pow(2, attempt) * 500))
      chat_with_retry(agent, input, attempt + 1)

    other ->
      other
  end
end
```

### RAG, end to end

```elixir
embedder = ExAgent.provider!(:embed)

# Index: embed documents, persist provenance next to each vector
{:ok, result} = ExAgent.embed(embedder, chunks, task: :retrieval_document)

rows =
  Enum.zip(chunks, result.vectors)
  |> Enum.map(fn {text, vec} ->
    %{text: text, embedding: vec, model: result.model, task: result.task}
  end)

Repo.insert_all(Chunk, rows)

# Query: the *query* task, not the document task
{:ok, q} = ExAgent.embed(embedder, question, task: :retrieval_query)
[qv] = q.vectors

context =
  rows
  |> Enum.sort_by(&ExAgent.Embeddings.cosine_similarity(&1.embedding, qv), :desc)
  |> Enum.take(5)
  |> Enum.map_join("\n\n", & &1.text)

{:ok, agent} = ExAgent.start_agent(role: :chat)
{:ok, answer} = ExAgent.chat(agent, """
Answer using only this context. Say "I don't know" if it is not there.

#{context}

Question: #{question}
""")
```

### Multi-tenant — a key per request

```elixir
def answer_for(tenant, question) do
  provider = ExAgent.provider!(:chat, api_key: tenant.openai_key, model: tenant.model)
  {:ok, agent} = ExAgent.start_agent(provider: provider)

  try do
    ExAgent.chat(agent, question)
  after
    ExAgent.stop_agent(agent)
  end
end
```

### Vision on a self-hosted container

```elixir
# config/runtime.exs
config :ex_agent, :roles,
  vision: {ExAgent.Providers.OpenAICompatible,
             base_url: System.fetch_env!("MODAL_URL") <> "/v1",
             model: "Qwen/Qwen3-VL-8B-Instruct",
             modalities: [:text, :image, :video],
             headers: [{"Modal-Key", System.fetch_env!("MODAL_KEY")},
                       {"Modal-Secret", System.fetch_env!("MODAL_SECRET")}]}
```

```elixir
# Images and video reach vLLM three ways: URL, local path (base64 data URI), raw bytes.
ExAgent.chat_with(:vision, "What is in this image?",
  files: [%{url: "https://cdn.example.com/x.jpg", mime_type: "image/jpeg"}])

ExAgent.chat_with(:vision, "Describe this video.",
  files: [%{path: "clip.mp4", fps: 1}])

# Two attachments in one turn become two content parts
ExAgent.chat_with(:vision, "Which of these is a chart?",
  files: [%{data: png_bytes, mime_type: "image/png"},
          %{url: "https://cdn.example.com/chart.png"}])
```

### Streaming into Phoenix LiveView

```elixir
def handle_event("ask", %{"q" => q}, socket) do
  parent = self()

  Task.start(fn ->
    socket.assigns.agent
    |> ExAgent.chat_stream(q)
    |> Enum.each(fn
      %ExAgent.Chunk{type: :text_delta, text: t} -> send(parent, {:delta, t})
      %ExAgent.Chunk{type: :done, error: nil} -> send(parent, :done)
      %ExAgent.Chunk{type: :done, error: e} -> send(parent, {:failed, e})
      _ -> :ok
    end)
  end)

  {:noreply, assign(socket, answer: "", streaming: true)}
end

def handle_info({:delta, t}, socket),
  do: {:noreply, assign(socket, answer: socket.assigns.answer <> t)}

def handle_info(:done, socket), do: {:noreply, assign(socket, streaming: false)}

def handle_info({:failed, error}, socket),
  do: {:noreply, socket |> put_flash(:error, Exception.message(error)) |> assign(streaming: false)}
```

### Testing without network

Point a role at a stubbed provider and application code needs no injection plumbing.

```elixir
# config/test.exs
config :ex_agent, :roles, chat: {MyApp.StubProvider, replies: ["canned reply"]}
```

Or stub the HTTP layer of a real provider — every provider exposes its `Req` client.
`Req.new(plug: ...)` needs `{:plug, "~> 1.0", only: :test}` in your deps:

```elixir
provider = %ExAgent.Providers.OpenAI{
  api_key: "sk-test", model: "gpt-4o",
  req: Req.new(plug: fn conn ->
    Req.Test.json(conn, %{"choices" => [
      %{"message" => %{"role" => "assistant", "content" => "stubbed"}}
    ]})
  end)
}
```

---

## Custom providers

Implement `ExAgent.Provider`. Only `chat/3` is required; `upload/4`, `stream/3`,
`embed/3`, and `supported_modalities/1` are optional. `ExAgent.Error.from_result/2` does
the status classification and keeps the original body in `:raw`.

```elixir
defmodule MyApp.Providers.Anthropic do
  @behaviour ExAgent.Provider

  defstruct [:api_key, :req, model: "claude-sonnet-4-5", tools: [], system_prompt: nil]

  def new(opts) do
    provider = struct!(__MODULE__, opts)
    %{provider | req: Req.new(
        base_url: "https://api.anthropic.com/v1",
        headers: [{"x-api-key", provider.api_key}, {"anthropic-version", "2023-06-01"}]
      )}
  end

  @impl true
  def chat(provider, messages, _opts) do
    body = %{
      "model" => provider.model,
      "max_tokens" => 1024,
      "messages" => Enum.map(messages, &%{"role" => to_string(&1.role), "content" => &1.content})
    }

    Req.post(provider.req, url: "/messages", json: body)
    |> ExAgent.Error.from_result(__MODULE__)
    |> case do
      {:ok, %{"content" => [%{"text" => text} | _]}} ->
        {:ok, msg} = ExAgent.Message.new(role: :assistant, content: text)
        {:ok, ExAgent.Response.new(msg)}

      {:ok, body} ->
        {:error, ExAgent.Error.unexpected_response(body, __MODULE__)}

      {:error, _error} = failure ->
        failure
    end
  end

  @impl true
  def supported_modalities(_provider), do: [:text, :image, :document]
end

provider = MyApp.Providers.Anthropic.new(api_key: "sk-ant-...")
{:ok, agent} = ExAgent.start_agent(provider: provider)
```

`chat/3` returns `{:ok, %ExAgent.Response{}}`, `{:tool_calls, calls}`, or
`{:error, %ExAgent.Error{}}`. Each call is `%{"name" => name, "args" => args}`, plus
`"id"` where the provider issues one — models request several tools per turn, so it is a
list. `{:tool_call, name, args}` is still accepted for a single call.

**Each provider owns its own rules.** The dispatcher stays generic — it reads
`supported_modalities/1` rather than knowing which vendor accepts video, and the modality
gate, error classification, and SSE framing are shared because they are genuinely
provider-independent. Anything vendor-specific lives in that provider's module or service:

| Rule | Where it lives |
|---|---|
| Which modalities are accepted | `:modalities` on the provider struct |
| Inline ceiling and upload behaviour | that provider's service |
| Embedding task vocabulary | that provider's embed service |
| Content-part shapes | that provider's `format_attachment/1` |

So a new provider is free to disagree. If yours implements `upload/4`, build its
references with your own provider atom — `ExAgent.FileRef` requires only that a reference
carries a `:file_id` or a `:file_uri`:

```elixir
{:ok, ref} = ExAgent.FileRef.new(
  provider: :my_llm,
  file_uri: "https://my-llm.test/files/abc",
  mime_type: "application/pdf"
)
```

## Architecture

```
Application (ex_agent)
  ExAgent.AgentSupervisor (:one_for_one)
    ├── ExAgent.UploadCache               owns a public named ETS table
    ├── ExAgent.AgentDynamicSupervisor    runtime agents
    └── ExAgent.TaskSupervisor            async chat, parallel subagents, router dispatch
```

- **Behaviour dispatch** — `ExAgent.Provider` is both the contract and the dispatcher,
  resolving the callback module from the struct
- **Thin providers** — callbacks delegate to service modules; HTTP stays out of providers
- **Modality gate in the dispatcher** — one place covers `chat/3` and `stream/3` for every
  provider, instead of repeating the check per service
- **Public ETS for the upload cache** — read directly by concurrent agents; routing through
  the owning GenServer would serialize every read behind one process
- **Tool loop off the GenServer** — it runs in a supervised Task so the agent stays
  responsive to reads, but one turn at a time so context cannot race
- **Subagents bypass the GenServer** — ephemeral stateless calls use `Provider.chat/3`
- **Handoff returns to the caller** — agents stay decoupled; the caller routes
- **Roles cache in `:persistent_term`** — free reads; writes only at boot, since each one
  triggers a global GC scan

## Testing

```bash
mix test    # full suite, fully mocked — no network, no credentials
```

`test/ex_agent/live_api_test.exs` runs against **real provider APIs**. It is tagged
`:external` and excluded by default. Sections whose credentials are missing are skipped
with a reason, so a partial setup reports what it did not cover.

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export EX_AGENT_COMPAT_BASE_URL=https://your-app.modal.run/v1
export EX_AGENT_COMPAT_MODEL=Qwen/Qwen3-VL-8B-Instruct

mix test --only external                # everything you have keys for
mix test --only external --only gemini  # also: openai, compat, roles, patterns
```

That file is the only place where request shapes, field names, size thresholds, and enum
values are checked against what providers actually accept; every other test mocks HTTP.
Its `@moduledoc` documents every optional variable.

## License

Apache-2.0
