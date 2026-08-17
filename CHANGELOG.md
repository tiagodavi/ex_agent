## v0.4.1 (2026-08-17)

### Fixed

- **`:receive_timeout` is no longer hardcoded.** Every service pinned its HTTP timeout at
  5 minutes, and the chat services filtered call options through their own schema, so a
  `receive_timeout:` passed by a caller was dropped without a word. A call wrapped in a
  shorter supervising timeout, say a `Task` killed at 180 s, could not make the HTTP layer
  give up first: a request still willing to wait 300 s was reported as a dropped call at
  180 s even though the answer would have arrived. Uploads were worse off, running on Req's
  own short default, which a multi-megabyte video body cannot finish inside.

### Added

- **`:receive_timeout` on every provider**, defaulting to the previous 5 minutes:

      provider = OpenAICompatible.new(base_url: url, model: "Qwen3-VL", receive_timeout: 150_000)

  and overridable per request, where the timeout usually belongs, since a video call and a
  text call against the same endpoint do not deserve the same budget:

      ExAgent.chat(agent, "Describe this clip",
        files: [%{path: "clip.mp4"}],
        receive_timeout: :timer.seconds(150))

  Accepted by `chat/3`, `chat_stream/3`, `embed/3` and `rerank/4` across `Providers.OpenAI`,
  `Providers.Gemini`, `Providers.OpenAICompatible`, `Providers.JinaV5` and
  `Providers.JinaRerankerM0`. The connect timeout follows the same value, so a hung connect
  cannot outlive the ceiling you set, and attachment uploads inherit it too.

## v0.4.0 (2026-08-17)

### Added

- **Structured output.** Describe the shape you want with an ordinary `Ecto` embedded schema
  and get a struct back, cast and typed:

      {:ok, response} = ExAgent.chat(agent, "Extract the invoice",
                          files: [%{path: "inv.pdf"}], schema: Invoice)

      response.structured   #=> %Invoice{total: 128.4, currency: :EUR, issued_on: ~D[2026-03-14]}

  There is no ExAgent schema language and no macro: `use Ecto.Schema` is the whole contract.
  `:schema` works on `ExAgent.chat/3`, `chat_async/3`, `chat_stream/3` and `chat_with/3`,
  and `schema: {:list, Invoice}` returns a list. `:schema_doc` adds prose guidance, which
  becomes the schema root's description rather than a message, so it never enters history.

  Supported on `Providers.OpenAI` (`response_format` with `strict: true`),
  `Providers.Gemini` (`responseSchema` plus `responseMimeType`), and
  `Providers.OpenAICompatible`. Backed by a new optional
  `c:ExAgent.Provider.supports_structured_output?/1`: a provider that has not opted in
  refuses a `:schema` rather than quietly answering with prose.

  `Ecto` is an **optional** dependency, needed only for this.

- **`ExAgent.Schema`**, the reflection layer, usable on its own: `to_json_schema/2` and
  `cast/2`. Covers strings, integers, floats, decimals, booleans, dates, times, all four
  datetime variants, `Ecto.UUID`, `{:array, inner}`, `Ecto.Enum`, `embeds_one` and
  `embeds_many`, recursively.

- **`%ExAgent.Response{structured: ...}`** holds the cast struct or list, `nil` when no
  schema was given. `:content` still carries the raw JSON, so nothing is lost.

- **Typed tool parameters.** `:parameters` now accepts an `Ecto` schema module alongside a
  raw JSON Schema map, and the tool function receives a struct:

      ExAgent.Tool.new(
        name: "order_status",
        description: "Look up the delivery status of an order",
        parameters: OrderQuery,
        function: fn %OrderQuery{order_id: id} -> Repo.get(Order, id) end
      )

  A typo in the function body becomes a compile-time warning instead of a `KeyError` inside
  the tool loop. Arguments the model gets wrong are fed back to it as a tool error, so it can
  correct itself, and your function never runs on bad input.

- **`:retain_attachments` on `start_agent/1`.** Provider APIs are stateless, so the whole
  history goes out on every request and **every attachment from every earlier turn goes with
  it**, base64-encoded again each time. Media caps are per *request*, so an agent sending one
  image per turn eventually trips a limit nobody knowingly exceeded, and passing the same
  file twice puts the identical bytes in one request twice.

  `retain_attachments: false` sends only the newest turn's attachments. History is untouched,
  so `get_context/1` still records what was sent. The default stays `true`, because it is the
  only correct answer for a conversation that refers back to a file.

- **`ExAgent.collect/2`** takes `:schema` to cast a finished stream. It has to be given to
  both `chat_stream/3` and `collect/2`, because a stream is a plain enumerable and carries no
  memory of how it was built. There are no partial objects mid-stream.

- **`ExAgent.Error` types `:invalid_response` and `:refusal`.** Neither is retryable:
  under constrained decoding a mismatch means the endpoint is not honouring the schema, and
  resending a request the model declined will not change its mind.

### Changed

- **A model refusal is no longer classified `:server`.** An OpenAI message carrying
  `refusal` produced `{:error, %ExAgent.Error{type: :server, retryable?: true}}`, so a retry
  wrapper would resend a request the model had explicitly declined. It is now
  `{:error, %ExAgent.Error{type: :refusal, retryable?: false}}`.

  **This is visible to anyone matching on `%ExAgent.Error{type: :server}`.** A message with
  neither content, tool calls, nor a refusal is still `:server`.

### Security

- **`:req` floor raised from `~> 0.5` to `~> 0.7`.** Every version below 0.6.1 carries
  EEF-CVE-2026-49755 (HIGH), an unbounded-decompression denial of service on auto-decoded
  response bodies, and there is no 0.5.x patch. The 0.5 series also carries
  EEF-CVE-2026-49756 (LOW), multipart header injection through unescaped filenames, which
  reaches `ExAgent.upload_file/4`.

  This raises the requirement for consumers. The full live provider suite was run against
  both versions to confirm the upgrade changes no behaviour: 0.7.2 produced *fewer* failures
  than 0.5.17, and the remainder were provider-side rate limits in both.

## v0.3.0 (2026-08-11)

### Changed

- **Embedding task vocabularies now belong to each provider.** There was a single
  normalized set of eight atoms translated per provider, plus a `:task_map` to retarget the
  strings and a verbatim-string escape hatch. That model does not survive contact with real
  endpoints: Gemini's `taskType` is a closed enum of eight, Jina v5 has four task names plus
  a separate `prompt_name`, and OpenAI has no task field at all. Translating between them
  meant either dropping distinctions a model makes or inventing ones it does not - and the
  built-in map was already wrong, mapping `:retrieval_document` to Jina v3's
  `"retrieval.passage"`, which v5 removed.

  Each provider now declares its own atoms and rejects anything outside them.
  `ExAgent.embedding_tasks/1` lists them, backed by a new optional
  `c:ExAgent.Provider.embedding_tasks/1` callback.

  **Removed** from `ExAgent.Embeddings`: `tasks/0`, `valid_task?/1`, the `task_input` type,
  the `:task_map` option, and verbatim string tasks. A string is now rejected everywhere - an endpoint that does not recognize a task
  string answers 200 and leaves quietly wrong vectors in an index, so there is no safe
  version of "send it and hope".

  Gemini's own atoms are unchanged, so Gemini callers are unaffected.

- **`ExAgent.Providers.OpenAICompatible` no longer supports embeddings.** "Any endpoint
  speaking the OpenAI dialect" cannot have a task vocabulary, which is exactly what the
  removed `:task_map` was trying to paper over. the `OpenAICompatibleEmbedService` module
  is gone; the provider is chat-only. Use `ExAgent.Providers.JinaV5`, or a provider of your
  own, for embeddings against a self-hosted model.

### Fixed

Found in a pre-release audit of the code added for this version.

- **`ExAgent.Reranking.above/2` kept unscored results.** In Elixir's term ordering every
  atom sorts above every number, so `nil >= 0.5` is `true` and a result with no score
  survived any relevance floor. `above/2` now requires a numeric score, and the reranker
  service rejects the whole response if a result has no numeric `relevance_score`, so the
  situation cannot arise from a server that omits one.

- **`Reranking.take/2` returned `nil` for an out-of-range index**, which would put the
  string "nil" into a prompt. It now raises, naming the mismatch, and the service rejects
  a response whose indexes fall outside the documents that were sent.

- **A bad `:max_history` or `:max_tool_iterations` crashed the agent mid-turn.**
  `max_history: 0` was accepted by `start_link/1` and then raised a `FunctionClauseError`
  at the end of the first turn, pointing at the wrong line entirely. Both are validated
  when the agent starts.

- **`MapReduce` reported `{:error, :all_sections_failed}` with no reason.** The failures
  now travel with it, since "everything failed" alone cannot be debugged.

- **`Consensus` reported `{:error, :no_answers}` when nobody had been asked.** An empty
  `:voters` list or a non-positive `:samples` is now `{:error, :no_voters}`, distinct from
  every voter having been asked and failed, which returns the failures alongside.

- `MapReduce` and `Consensus` accumulated results with `++` per item, which is quadratic
  in the number of sections or voters. They prepend and reverse once.

### Changed

- **Pattern API is uniform.** Every workflow entry point is now `run/2` or `run/3`, and
  every builder that hands tools to an agent is `tools/1`. Three conventions had grown up
  side by side, and the `build_` prefix said nothing that the return type did not.

  | Before | Now |
  |---|---|
  | `Subagents.build_orchestrator_tools/1` | `Subagents.tools/1` |
  | `Subagents.invoke_subagents_parallel/2` | `Subagents.run/2` |
  | `Handoff.build_handoff_tool/3` | `Handoff.tools/1` |
  | `Handoff.execute_handoff/2` | `Handoff.run/2` |
  | `Router.route/2` | `Router.run/2` |
  | `Skills.evaluate_skills/2` | `Skills.evaluate/2` |

  `Handoff.tools/1` now takes a list of `%{name:, agent:, description:}` specs and returns
  a list, matching `Subagents.tools/1` exactly, so building several handoff targets is one
  call. `ExAgent.route/2` and `ExAgent.handoff/2` are unchanged.

- **The README is now a tutorial**, not a feature tour. Eleven steps from "ask one
  question" to a composed support pipeline, each a complete program you can paste into
  `iex -S mix`. Every block was executed against a live API before publishing, and the
  outputs shown are from real runs with a note that models vary.

  The step order is the teaching order: tool, then skill, then subagent, then handoff,
  followed by a table answering the question people actually have - *who answers the next
  message?* A subagent is a phone call you make while the customer waits; a handoff is
  passing the customer to a colleague.

  The handoff step now explains why `ExAgent.handoff/2` has to be called before talking to
  the target: it is what delivers the conversation, and without it the target agent is a
  stranger. Includes the before/after of what actually reaches the model, why the returned
  tuple is a proposal rather than a transfer, and why the async cast is not a race
  (Erlang orders messages between a pair of processes; measured 0 late arrivals in 200
  runs) along with the case where that guarantee does not hold.

- Em dashes removed from all documentation and source comments.

### Added

- **Four workflow patterns**, filling the gaps against the commonly documented
  catalogue (Anthropic's prompt chaining / routing / parallelization /
  orchestrator-workers / evaluator-optimizer, plus the sequential-workflow and
  reflection patterns that show up in every 2026 survey). ExAgent already had
  routing, orchestrator-workers, peer transfer, progressive disclosure, and the
  ReAct-style tool loop; these are the rest:

  - **`ExAgent.Patterns.Chain`** - a fixed sequence of steps, each working on the
    last one's output. Steps are plain functions, so validation, parsing, and
    database lookups sit in the line beside the LLM calls; `Chain.llm/2` builds an
    LLM step. A step returning `{:halt, value}` stops the line *without* it being a
    failure, which is how you decline to spend the remaining calls - and where a
    human approval gate belongs. Errors carry the failing step's index.

  - **`ExAgent.Patterns.Reflection`** - the evaluator-optimizer loop: draft,
    critique, revise, until the critic accepts or `max_rounds` (default 3) runs out.
    Exhausting the ceiling returns **`{:max_rounds, result}`**, not `{:ok, result}`:
    the last draft is there, but using unreviewed work has to be a choice rather
    than something handed over as if a reviewer had passed it. An LLM critic can
    always find something to complain about, so the ceiling is the difference
    between a workflow and a runaway bill.

  - **`ExAgent.Patterns.MapReduce`** - parallelization by sectioning: split an
    oversized input, process the pieces concurrently, combine them with either a
    function or another model (`reduce: {target, prompt_builder}`). One failing
    section does not fail the run; the reducer sees what survived and `:failures`
    reports the rest, because a summary of 38 of 40 interviews is worth having but
    not worth mistaking for all 40.

  - **`ExAgent.Patterns.Consensus`** - parallelization by voting: ask several times
    (or several models) and go with the answer that recurs. `:agreement` is the
    winner's share, which is the actual product - a low number is the signal to
    escalate rather than proceed. Ties resolve to the earliest answer
    deterministically, which needs care because `Enum.frequencies/1` returns a map
    and a map has no insertion order to fall back on.

  All four accept either a provider struct (stateless, no process) or a running
  agent (remembers the conversation) wherever they take a target - previously
  `Router` took only agents and `Subagents` only provider structs, and neither
  could be handed the other.

  Deliberately **not** added: plan-and-execute, which needs an LLM-authored plan
  parsed into executable steps and is brittle in exactly the way the rest of this
  library tries not to be - compose it from `Chain` and `Subagents` instead; and
  blackboard/swarm topologies, which the production write-ups consistently report
  losing to hierarchical and graph shapes.

### Changed

- The README's pattern section is now a **guide to choosing one**, not a feature
  list: analogies for all eight, a table keyed on when to reach for each, and
  worked comparisons of the pairs people conflate - Handoff vs Subagents (a lookup
  versus a transfer, settled by "who is the user talking to now?"), Skills vs
  Subagents (continuity versus isolation), and Reflection vs Consensus (sloppy work
  versus wrong work).

- **Reranking.** `ExAgent.rerank/4` and `ExAgent.rerank_with/4`, backed by a new optional
  `c:ExAgent.Provider.rerank/4` callback, returning an `ExAgent.Reranking` struct. Retrieval's
  second stage: embeddings compare independently computed vectors, which is what makes
  searching a corpus feasible, while a cross-encoder reads the query and one document
  together - more accurate, and far too slow to run over everything.

  `:index` is the contract, pointing back into the list you passed, so results map onto your
  own records without the server echoing text back. `ExAgent.Reranking.take/2` reorders a
  list; `above/2` applies a relevance floor, because ranking always returns *something* -
  the best of an irrelevant set still sorts first, and a floor is how you decline to answer.
  Scores are model-scoped: higher is more relevant and that is the only guarantee.

  Providers without a reranking endpoint return
  `{:error, %ExAgent.Error{type: :unsupported}}`. Emits
  `[:ex_agent, :rerank, :start | :stop | :exception]` telemetry.

- **`ExAgent.Providers.JinaRerankerM0`** - a reranking-only provider for a self-hosted
  `jina-reranker-m0` server. `chat/3` returns `:unsupported`, and there is no `embed/3`: a
  reranker scores query/document *pairs* and has no single-text vector to give.

  `:base_url` is required and takes bearer auth plus arbitrary `:headers`, so Modal's proxy
  auth works. Empty document lists, non-string documents, batches over 512, a blank query,
  a non-positive `:top_n`, and unrecognized options are all rejected before the request -
  the server rejects unknown body fields outright, so a typo has to be caught client-side or
  it comes back as a validation blob.

  The wire contract - `POST {base_url}/v1/rerank` with `query`/`documents`/`top_n`/
  `return_documents`, answering `results` with `relevance_score` and `document.text` - was
  verified against a live deployment. `return_documents` defaults to `true` there and
  `false` here, since `:index` already identifies each document. This is **not** the shape of
  Jina's hosted `api.jina.ai/v1/rerank`, whose `documents` take `{"text": ...}` /
  `{"image": ...}` objects.

- **`ExAgent.Providers.JinaV5`** - an embeddings-only provider for a self-hosted Jina
  embeddings v5 server, with v5's own tasks: `:retrieval`, `:text_matching`, `:clustering`,
  `:classification`. v5 moved the query/document distinction *out* of the task and into a
  separate `prompt_name`, which is why the module is named for the version: v3 and v4
  spelled the same thing as a single `"retrieval.query"` / `"retrieval.passage"` task, so
  one module covering both would have to lie about one of them.

  `prompt_name` is required for `:retrieval` and rejected for the other tasks; Matryoshka
  truncation is validated against the trained widths (32, 64, 128, 256, 512, 768, 1024);
  batches are capped at 512 inputs. All three are rules the server enforces, checked
  client-side so the failure names the fix instead of arriving as a 400. The server owns
  normalization, so vectors are returned untouched - `args: [normalize: false]` really does
  give you a non-unit vector.

  `chat/3` returns `{:error, %ExAgent.Error{type: :unsupported}}` pointing at a chat
  provider. `:base_url` is required and takes bearer auth plus arbitrary `:headers`, so
  Modal's proxy auth works.

  The wire contract - `POST {base_url}/embed` with `texts`/`task`/`prompt_name`/
  `dimensions`/`normalize`, answering `embeddings` - was verified against a live deployment,
  not inferred from a model card. It is **not** the shape of Jina's hosted `api.jina.ai`
  service, which speaks an OpenAI-style `/v1/embeddings`.

- **`:args` on `embed/3`** - extra request-body parameters as a keyword list or map, for
  what this library does not model:

      ExAgent.embed(jina, chunks, task: :retrieval, args: [prompt_name: :document])
      ExAgent.embed(openai, chunks, args: [encoding_format: "base64"])

  Each provider validates keys **and** values against what its own endpoint accepts and
  rejects the rest, so `prompt_nane:` fails naming the accepted keys instead of being
  ignored by the server. Atoms are accepted where the endpoint wants one of a fixed set of
  strings. Gemini accepts no extra args and says so; OpenAI accepts `encoding_format` and
  `user`; Jina v5 accepts `prompt_name` and `normalize`, the only extra fields its server
  permits.

- `ExAgent.Embeddings.normalize_args/1`, for providers implementing the same option.

### Fixed

An end-to-end audit of the library found the following. Every one of them shipped with a
green suite: the tests covered the shape of each code path but not the behaviour a user
would observe. `test/ex_agent/regressions_test.exs` now covers each one.

- **Streaming stole messages from the caller's mailbox.** `chat_stream/3` runs in the
  *calling* process, and the SSE transport used a bare `receive` that matched anything;
  `Req.parse_message/2` answering `:unknown` then discarded it. Streaming inside a
  LiveView or GenServer silently ate that process's own messages, and the matching
  `handle_info` simply never fired. The receive is now selective on the response ref.

- **Gemini ignored `:max_tokens` entirely.** The option was merged under one key and read
  under another, so neither the provider setting nor a per-call override ever reached
  `generationConfig` and every response used the API default. Google's own
  `:max_output_tokens` spelling is accepted as an alias.

- **A Gemini reasoning part was returned as the answer.** Only the first content part was
  read, and a `thought` part matched the text clause - so the model's scratchpad became
  `:content` and the real answer was discarded. Text split across parts was truncated to
  the first piece for the same reason. Reasoning now lands in `:thinking`, as it already
  did when streaming.

- **Parallel tool calls were silently dropped.** Both dialects returned only the first
  call, so the model believed tools had run that never did. `c:ExAgent.Provider.chat/3`
  now answers `{:tool_calls, calls}` with every call; `{:tool_call, name, args}` is still
  accepted from providers written against the older contract.

- **Tool call ids were fabricated.** The assistant message was rebuilt with `id = name`,
  discarding the id the API issued - so calling one tool twice in a turn produced two
  colliding ids. The provider's id is now carried through, and a tool result correlates
  back by it (Gemini correlates by function name, which travels alongside).

- **A tool returning anything but a string crashed the turn.** `to_string/1` on a map
  raised `Protocol.UndefinedError`, killing the task and surfacing as an opaque `:server`
  error - for the most natural tool shape there is. Non-string results are now JSON
  encoded.

- **A skill never deactivated.** Applying one overwrote the provider's `system_prompt`
  permanently, so the first activation repainted the agent for the rest of its life - a
  "SQL expert" answering jokes. Skills are re-evaluated every turn and now restore the
  agent's own prompt when they stop matching.

- **One failing agent took down a whole Router run.** Only `{:exit, :timeout}` was
  handled, so any other crash raised a `CaseClauseError` in the caller, discarding the
  routes that had already answered. Both `Router` and `Subagents` now report a failure
  per route, named. A handoff result no longer falls through unmatched.

- **`parse_response/2` raised on a message with no `"content"` key.** A bare refusal gave
  a `CaseClauseError` instead of a normalized `{:error, %ExAgent.Error{}}`.

- **OpenAI could never reference an uploaded image.** Attachments over the inline ceiling
  were uploaded and then referenced as an `image_file` content part - which is the
  Assistants API's shape and which chat completions rejects outright. Verified against
  the live API: no file-id shape works for images there. Oversized images and image
  `file_refs` now return `{:error, %ExAgent.Error{type: :unsupported}}` with the fix in
  the message, instead of spending an upload on a request that would always fail.

- **API keys were printed by `inspect/1`.** No provider redacted its credential, so every
  crash report, `dbg`, and Logger metadata dump leaked it. All three providers now derive
  `Inspect` excluding `:api_key`, `:req`, and (for `OpenAICompatible`) `:headers`, where
  gateway credentials live.

- **Streaming with tools billed twice.** The tool loop ran non-streamed to completion and
  then *discarded the finished answer* to regenerate it as a stream - two full completions
  per streamed turn. Every turn is now streamed once, with tools run between turns; the
  consumer still sees exactly one terminal `:done` chunk.

- **SSE multi-line `data:` fields were concatenated without a separator.** The spec joins
  them with a newline. JSON payloads survived either way; a plain-text stream did not.

- Skills, subagents, and the streaming tool loop assumed every provider struct carries
  `:tools` and `:system_prompt`, reintroducing the `KeyError` already fixed on the chat
  path. All of them now populate a field only when the provider declares it.

- `ExAgent.Agent.chat/3`'s `@spec` still promised `{:ok, Message.t()}` after the switch to
  `ExAgent.Response`.

### Changed

- **`:max_tokens` and `:temperature` now default to `nil`** on every provider and are
  omitted from the request, so the model's own defaults apply. The previous
  `max_tokens: 512` truncated most real answers at `finish_reason: :length`, and
  `temperature: 0.6` broke models that reject the parameter outright (o-series,
  search-preview). **Set them explicitly if you relied on the old values.** Both now also
  accept an integer, where `temperature: 1` used to raise.

- OpenAI embeddings reject a batch over 2048 inputs up front, naming the limit, rather
  than letting the API 400.

### Added

- **`ExAgent.Telemetry`.** `[:ex_agent, :chat | :embed | :tool, :start | :stop | :exception]`
  events carrying duration, token counts, model, and - on failure - `:error_type` and
  `:retryable?`. A library that calls billed APIs has to be measurable; nothing is logged
  on your behalf. Adds a `:telemetry` dependency.

- **`:max_history` on `start_agent/1`** and `ExAgent.Context.trim/2`. History was
  unbounded, so every turn resent the whole transcript until the model returned
  `:context_length`. Opt-in, because silently forgetting what a user said is the caller's
  decision. Leading system messages survive the window, and a tool result is never
  orphaned from the assistant message that requested it.

- **`:max_tool_iterations` on `start_agent/1`**, replacing the hard-coded ceiling of 10.

### Added

- **Provider roles.** `config :ex_agent, :roles, chat: {Module, opts}` maps a purpose to
  a provider, and `ExAgent.provider!(:chat)` returns an ordinary provider struct usable
  anywhere a hand-built one is - `start_agent/1`, `ExAgent.Provider.chat/3`, subagent
  specs, every pattern. Role names are arbitrary atoms. `start_agent(role: :vision)` is
  shorthand; `:role` and `:provider` are mutually exclusive.

  Purely additive: every existing entry point still takes a struct, unchanged.

  `ExAgent.chat_with/3`, `stream_with/3` and `embed_with/3` are stateless one-shot
  wrappers that bypass the agent GenServer. They do **not** run the tool loop - a
  tool-configured provider returns the raw `{:tool_call, name, args}`.

  Roles resolve once at application start and cache in `:persistent_term`, so lookups
  cost nothing on the request path; per-call overrides (`provider!/2`) build a fresh
  struct rather than writing to the cache. A module that is missing, lacks `new/1`, does
  not implement `ExAgent.Provider`, or whose `new/1` raises fails the boot with the role
  name in the message, so a missing credential crashes at deploy time instead of on the
  first request. Option values may be a zero-arity function or `{m, f, a}`, resolved once
  at boot, for vault-backed credentials.

  Note that this is the first thing in `lib/` to read `Application.get_env` - the library
  was otherwise configured entirely through explicit structs, and still can be.

- **`ExAgent.Error`.** A normalized error struct returned by every provider operation,
  carrying `:type`, `:message`, `:status`, `:provider`, `:raw` and `:retryable?`. HTTP
  statuses are classified into one vocabulary (`:auth`, `:rate_limit`,
  `:context_length`, `:invalid_request`, `:not_found`, `:timeout`, `:server`,
  `:transport`, `:unsupported`), so retry logic is written once rather than per
  provider. `ExAgent.Error.from_result/2` does the classification for custom providers.
  The struct is also an exception, so it can be raised where no return value exists.

- **URL file sources.** `files: [%{url: "https://..."}]` hands the URL straight to the
  provider - Gemini as `file_data.file_uri`, OpenAI as `image_url` / `file_url`. ExAgent
  never fetches the URL, so no bytes cross your application.
- **Optional `:mime_type`.** Inferred from the file extension for `:path` and `:url`
  (query strings ignored) and from magic bytes for `:data` (PNG, JPEG, GIF, WebP, WAV,
  MP3, MP4/QuickTime/M4A, PDF). An explicit `:mime_type` still wins. When the type cannot
  be determined the call fails with an error naming `:mime_type` rather than guessing.
- **`ExAgent.Source`.** Pure MIME-inference and modality helpers, with no provider
  knowledge.
- **`ExAgent.Attachment`.** Attachments normalize into a struct carrying `:kind`,
  `:mime_type`, `:modality`, `:byte_size` and `:provider_opts` (video `:fps` /
  `:max_frames` are lifted into the latter).

- **Modality gating.** The optional `c:ExAgent.Provider.supported_modalities/1` callback
  declares which attachment modalities a provider accepts (`:image`, `:document`,
  `:video`, `:audio`). `ExAgent.Provider.chat/3` and `stream/3` check every attachment
  against it before building a request. Providers that omit the callback are text-only,
  so an unsupported attachment fails loudly instead of being dropped.

  Every built-in provider takes a `:modalities` option, because modality support is a
  property of the *model*, not the vendor - `o1-mini` reads no images. Defaults are
  `[:text, :image, :document]` for OpenAI, those plus `:video` and `:audio` for Gemini,
  and `[:text]` for `OpenAICompatible` (one container serves one model). Narrowing makes
  the gate fire locally instead of letting the provider 400 later:

      OpenAI.new(api_key: key, model: "o1-mini", modalities: [:text])

  ExAgent keeps no model-to-modality table on purpose: it would go stale silently, and
  whoever picked the model already knows.

- **Embeddings.** `ExAgent.embed(provider, inputs, opts)` returns an
  `%ExAgent.Embeddings{}` carrying `vectors`, `model`, `provider`, `dimensions`, `task`,
  and `usage`. It takes a provider struct rather than an agent pid - embedding is
  stateless. Backed by a new optional `c:ExAgent.Provider.embed/3` callback; providers
  without an embeddings endpoint return
  `{:error, %ExAgent.Error{type: :unsupported}}`.

  One normalized task vocabulary (`:retrieval_query`, `:retrieval_document`,
  `:similarity`, `:classification`, `:clustering`, `:question_answering`,
  `:fact_verification`, `:code_query`) is translated per provider:
  `gemini-embedding-001` takes a `taskType` enum, `gemini-embedding-2` has no such field
  and takes a text prefix (with an optional `:title` per input), OpenAI has no task
  support and **errors** rather than dropping it, and OpenAI-compatible endpoints take a
  `task` body field whose strings are overridable via `:task_map`.

  Notable correctness details: `:model` is always resolved to an embedding model and never
  the provider's chat model; Gemini requests always use `batchEmbedContents` with one
  `Content` per input, because a flat list returns a single *aggregated* vector on
  `gemini-embedding-2`; OpenAI responses are re-sorted by `index`, which the API does not
  guarantee; and truncated `gemini-embedding-001` vectors are L2-normalized client-side,
  which that model does not do for you. An unknown Gemini embedding model errors rather
  than guessing a family - pass `:embedding_family` to adopt a newer one.

  On `OpenAICompatible` only, `:task` also accepts a **raw string**, sent verbatim with
  no translation and no validation - a self-hosted endpoint serves whatever model you
  deployed, and those vocabularies change between versions (Jina v3's
  `"retrieval.passage"` became a `"retrieval"` task plus a prompt in v5, which also added
  `"text-matching"`). Gemini and OpenAI take atoms only: `taskType` is a closed enum and
  OpenAI has no task field, so a string there is a typo far more often than a new value
  and is rejected naming the valid atoms. Atoms stay validated everywhere, and the result
  carries back exactly what was passed.

  `ExAgent.Embeddings` also exposes `tasks/0`, `valid_task?/1`, `l2_normalize/1`, and
  `cosine_similarity/2`. **Persist `model`, `dimensions`, and `task` alongside every
  vector** - embedding spaces are model-scoped and mixing them degrades retrieval
  silently.
- **`ExAgent.Providers.OpenAICompatible`.** One provider for any endpoint speaking the
  OpenAI chat-completions dialect - self-hosted vLLM (including behind Modal),
  OpenRouter, Together, Groq. Takes arbitrary `:headers` (so Modal's `Modal-Key` /
  `Modal-Secret` proxy auth works, where `ExAgent.Providers.OpenAI` had no way to set
  them), with `:api_key` as sugar for a bearer header that explicit headers override.
  `:modalities` is declared per deployment and defaults to `[:text]`. Media is carried in
  `image_url` / `video_url` / `audio_url` content parts whose URL may be a `data:` URI.
  There is no Files API, so an attachment past `:max_inline_bytes` (32 MB default)
  returns `{:error, %ExAgent.Error{type: :unsupported}}` rather than being truncated.
  `probe/1` checks that the endpoint actually serves the configured model.
- **Shared OpenAI dialect helper** (`lib/ex_agent/services/openai_dialect.ex`, internal).
  Holds the request/response
  shaping shared by every dialect speaker, so the OpenAI and OpenAI-compatible services
  no longer duplicate it.
- **Adaptive inline-vs-upload.** Both services now choose how each attachment is
  delivered rather than leaving it to the caller: URLs and existing `FileRef`s are
  referenced as-is, bytes under the inline ceiling are base64-encoded, and anything
  larger is uploaded through the provider's Files API and referenced. Gemini inlines up
  to 20 MB (50 MB for `application/pdf`) and references by URI; OpenAI inlines up to
  20 MB and references by `file_id`. Uploads reuse the provider's `Req` client and are
  deduplicated through `ExAgent.UploadCache`; `upload_cache: false` on the provider opts
  out.
- **Video options.** `:fps` maps to Gemini's `video_metadata`, and string-keyed
  `:provider_opts` entries are merged into the media part verbatim. The
  `OpenAICompatible` `video_url` content part and `:fps` passthrough are verified against
  a live vLLM (Qwen3-VL) deployment, from both an http URL and a base64 data URI.
- **Gemini upload polling is configurable.** `:poll_interval_ms` and
  `:max_poll_attempts` are now options, defaulting to `2_000` / `60` (~2 minutes, up
  from ~10 seconds) - large files and video need it.
- **`ExAgent.UploadCache`.** An ETS-backed cache that lets the same bytes reuse an
  existing `ExAgent.FileRef` instead of re-uploading. Entries are keyed by
  `{scope, sha256(bytes)}` where the scope digests the provider module, base URL, and
  API key - so two accounts never share a file reference, and the key itself is never
  stored. An expired `FileRef` is treated as a miss and evicted. Added to the
  supervision tree ahead of the agent supervisor; `clear/0` empties it.

### Fixed

- **A failed turn poisoned the agent.** The user message was committed to context even
  when the turn failed, so a message the provider had already refused - a rejected
  attachment, say - was resent on every later turn and every one of them failed. "Fails
  loudly" became "fails forever". A failed turn now leaves no trace, which also stops a
  retry after a transient 429 from duplicating the question in history.

- **`chat_stream/3` raised on a rejected attachment.** The modality gate raises inside
  `ExAgent.Provider.stream/3` because a lazy enumerable has nowhere to carry an error at
  construction time, and that escaped to the consumer - contradicting the documented
  promise that streaming never raises, and leaving the agent stuck in `:processing`. It
  now arrives as the terminal `:done` chunk like every other stream failure, and the
  agent is released.

- **The agent required a `:tools` field on every provider struct.** `run_tool_loop/3`
  and the streaming path both did `%{provider | tools: ...}`, so a provider without tool
  support crashed with a `KeyError` that surfaced as an opaque `{:error, %Error{type:
  :server}}`. The field is now populated only when the provider declares one.

- **`ExAgent.FileRef` rejected custom providers.** `:provider` was validated against a
  hardcoded `[:openai, :gemini]`, so a third-party provider implementing the optional
  `c:ExAgent.Provider.upload/4` callback could not build the reference its own callback
  has to return. The built-in services construct `%FileRef{}` structs directly and never
  called `new/1`, so the closed list protected nothing - it only walled out everyone
  else. Any atom is now accepted, and a reference need only carry a `:file_id` or a
  `:file_uri`; OpenAI's and Gemini's specific field requirements still apply to them,
  since their services pattern-match on those fields.

- **`OpenAICompatible` shaped documents as images.** `:document` was declarable through
  `:modalities` but `format_attachment/1` fell through to `image_url`, so a PDF was sent
  as an image part and the gateway either rejected it or read nothing. Documents now use
  the dialect's `file` part - `file_data` for bytes (with the required `filename`) and
  `file_url` for a URL - which is what a gateway fronting a document-reading model
  expects. The moduledoc previously claimed documents were unsupported; a model behind
  OpenRouter or Modal may well read them, so it is a deployment property like every other
  modality.

- **Gemini streaming produced no text at all.** Gemini terminates SSE events with CRLF,
  but `ExAgent.SSE.take_events/1` split only on `"\n\n"`. A CRLF stream contains no such
  boundary, so every frame stayed buffered, no frame was ever decoded, and the stream
  ended with its terminal chunk and empty content - silently, with no error. Framing now
  accepts CRLF, LF, and bare CR per the SSE spec, on both event and line boundaries.
  Found by running against the live API; every mocked test hand-wrote LF bodies.

- **Streams ended with two `:done` chunks whenever the model reported a finish reason.**
  Provider mappers turn a finish-reason frame into a `:done` chunk and the transport
  appends the terminal one. Since real responses always finish, the documented
  "exactly one `:done` chunk" invariant was broken in practice for every provider. The
  transport now consumes the mapper's `:done` for its finish reason and emits the single
  terminal chunk itself.

- **A tool returning a bare value crashed the tool loop.** `ExAgent.Tool`'s `:function` is
  typed `(map() -> any())` and its own doctest returns a bare `:ok`, but the agent matched
  only `{:ok, _}` / `{:error, _}` / `{:handoff, _, _}` - anything else raised
  `CaseClauseError` inside the supervised task, surfacing as an opaque `:server` error. An
  unwrapped return is now taken as the result.

- **Documented that OpenAI's `:web_search` needs `temperature: nil`.** `web_search_options`
  is only accepted by a `*-search-preview` model, and those reject `temperature` - so the
  provider's own `temperature: 0.6` default made the shipped example return HTTP 400.
  ExAgent still forwards `temperature` as configured rather than dropping it when a model
  objects; a silently ignored sampling parameter is worse than a 400 naming the field.

- **Retired Gemini default model.** `gemini-2.0-flash` is no longer served and returns
  HTTP 429 with a zero free-tier quota rather than a clear 404. The default is now
  `gemini-3.6-flash`. Pass `:model` explicitly to pin a different one.

- **A malformed attachment no longer crashes the agent.** `ExAgent.chat/3` and
  `ExAgent.chat_stream/3` matched on `{:ok, msg} = Message.new(...)`, so an unreadable
  path turned into a `MatchError` inside `handle_call` and took the agent process down.
  Both now return `{:error, %ExAgent.Error{type: :invalid_request}}`.

- **Normalized stream chunks.** `ExAgent.chat_stream/3` and `ExAgent.Provider.stream/3`
  now yield `%ExAgent.Chunk{}` structs instead of bare strings, surfacing what streaming
  previously discarded: reasoning traces (`:thinking_delta`), tool-call deltas, token
  usage, and finish reasons. Every stream ends with exactly one `:done` chunk.
- **`ExAgent.collect/1`.** Folds a chunk stream into the same `ExAgent.Response` that
  `chat/3` returns, reassembling fragmented tool-call arguments by index, so streaming
  and non-streaming share one downstream code path.
- **`ExAgent.SSE`.** Server-Sent Events framing extracted from the streaming transport
  and made public, with `decode/1` returning complete frames plus the unconsumed
  remainder. A frame split across TCP reads is reassembled correctly - now covered by a
  test that splits a body at every byte boundary.

### Removed (breaking)

- **`ExAgent.Providers.DeepSeek`.** DeepSeek speaks the OpenAI chat-completions dialect,
  so `ExAgent.Providers.OpenAICompatible` covers it with no loss of capability:

      # before
      ExAgent.Providers.DeepSeek.new(api_key: key, model: "deepseek-reasoner")

      # after
      ExAgent.Providers.OpenAICompatible.new(
        base_url: "https://api.deepseek.com/v1",
        api_key: key,
        model: "deepseek-reasoner"
      )

  Reasoning traces still arrive as `:thinking_delta` chunks - the `reasoning_content`
  field is handled by the shared dialect, not by the removed module. The built-in
  `:thinking` tool went with it; pick the reasoner model instead.

### Changed (breaking)

- **`chat/3` returns `%ExAgent.Response{}`.** Previously `{:ok, %ExAgent.Message{}}`.
  The response carries `:content`, `:usage`, `:finish_reason`, `:tool_calls`,
  `:thinking`, and the `:message` appended to conversation history:

      # before
      {:ok, %ExAgent.Message{content: content}} = ExAgent.chat(agent, "Hi")

      # after
      {:ok, %ExAgent.Response{content: content}} = ExAgent.chat(agent, "Hi")
      # ...or response.message for the Message struct itself

  This cascades through the `c:ExAgent.Provider.chat/3` callback, so custom providers
  must return a `Response` - build one with `ExAgent.Response.new/2`.
- **Streams yield `%ExAgent.Chunk{}` instead of `String.t()`.**

      # before
      agent |> ExAgent.chat_stream("Hi") |> Enum.each(&IO.write/1)

      # after
      agent
      |> ExAgent.chat_stream("Hi")
      |> Enum.each(fn
        %ExAgent.Chunk{type: :text_delta, text: text} -> IO.write(text)
        _chunk -> :ok
      end)

      # ...or collect it into a single response
      {:ok, response} = agent |> ExAgent.chat_stream("Hi") |> ExAgent.collect()

- **`ExAgent.StreamError` is removed, and streaming never raises.** A non-200 response, a
  transport failure, a busy agent, and an idle timeout all arrive as a terminal `:done`
  chunk carrying an `ExAgent.Error`. Previously `chat_stream/3` raised *eagerly* for a
  busy agent but *lazily* for an HTTP error, forcing consumers to wrap both the call site
  and the consumption site in `try`. Partial output already emitted stays valid.
- **A stream idling for 5 minutes now reports a timeout** instead of halting silently,
  which was indistinguishable from clean completion.

- **DeepSeek attachments.** Previously `raise`d `ArgumentError` from inside the service
  (and the README claimed they were silently ignored - neither was right). Attaching a
  file to DeepSeek now returns `{:error, %ExAgent.Error{type: :unsupported}}` before the
  request is built.
- **`:path` attachments are read lazily.** `Message.new/1` now records only the file's
  size; the bytes are read when the request is built. This keeps a large file out of
  conversation history, where it would otherwise be re-encoded on every turn. The
  behaviour change: a file deleted between attaching and sending now fails at send time
  rather than at attach time.
- **Attachment element type.** `Message.attachments` now holds `%ExAgent.Attachment{}`
  structs rather than bare maps. Since a struct is a map, code matching on
  `%{data: data, mime_type: mime_type}` or `%{file_ref: %ExAgent.FileRef{}}` keeps
  working; code using `Map.keys/1`, exact-map patterns, or `Map.get/3` defaults on
  optional keys (a struct key is present-but-`nil`, so the default never applies) needs
  updating.
- **Error shape.** All providers, services and upload services now return
  `{:error, %ExAgent.Error{}}` instead of `{:error, {status, body}}`. The original
  body is preserved in `:raw`, so the migration is mechanical:

      # before
      {:error, {status, body}} -> handle(status, body)

      # after
      {:error, %ExAgent.Error{status: status, raw: body}} -> handle(status, body)

  Replaced along with it: `{:error, {:unexpected_response, body}}` and
  `{:error, {:unexpected_parts, parts}}` are now `%ExAgent.Error{type: :server}` with
  the offending payload in `:raw`; `{:error, {:unsupported, :upload, module}}` is now
  `%ExAgent.Error{type: :unsupported}`; the Gemini upload atoms
  `:file_processing_timeout` and `:file_processing_failed` are now
  `%ExAgent.Error{type: :timeout}` and `%ExAgent.Error{type: :server}`; and
  `ExAgent.upload_file/4` no longer leaks a bare posix atom for an unreadable path.
- **`ExAgent.Provider.stream/3`** raises `ExAgent.Error` with `type: :unsupported`
  instead of `ArgumentError` when a provider does not implement `stream/3`.

## v0.2.0 (2026-07-20)

### Added

- **Streaming.** `ExAgent.chat_stream/3` (agent-level) and `ExAgent.Provider.stream/3`
  (provider-level) return a lazy `Stream` of text chunks. Tool-call turns are
  resolved non-streamed; only the final assistant turn is streamed. All three
  providers implement the optional `stream/3` callback.
  Raises `ExAgent.StreamError` on non-200 responses / when the agent is busy.

### Changed

- **Providers are now a behaviour instead of protocols.** The `ExAgent.LlmProvider`
  and `ExAgent.FileUploader` protocols were removed and replaced by a single
  `ExAgent.Provider` behaviour (`chat/3` required, `upload/4` optional). Custom
  providers now declare `@behaviour ExAgent.Provider` and implement `chat/3`
  (and optionally `upload/4`) as public functions instead of using `defimpl`.
- **Non-blocking agent.** `ExAgent.Agent` now runs the tool loop off the GenServer
  (via a supervised task), so an agent stays responsive to reads (`get_context`)
  and casts while a request is in flight. A concurrent `chat/3` on a busy agent
  now returns `{:error, :busy}` instead of serializing behind the mailbox.

## v0.1.0 (2026-03-30)

First release!