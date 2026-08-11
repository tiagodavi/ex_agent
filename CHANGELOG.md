## v0.3.0 (unreleased)

### Added

- **Provider roles.** `config :ex_agent, :roles, chat: {Module, opts}` maps a purpose to
  a provider, and `ExAgent.provider!(:chat)` returns an ordinary provider struct usable
  anywhere a hand-built one is — `start_agent/1`, `ExAgent.Provider.chat/3`, subagent
  specs, every pattern. Role names are arbitrary atoms. `start_agent(role: :vision)` is
  shorthand; `:role` and `:provider` are mutually exclusive.

  Purely additive: every existing entry point still takes a struct, unchanged.

  `ExAgent.chat_with/3`, `stream_with/3` and `embed_with/3` are stateless one-shot
  wrappers that bypass the agent GenServer. They do **not** run the tool loop — a
  tool-configured provider returns the raw `{:tool_call, name, args}`.

  Roles resolve once at application start and cache in `:persistent_term`, so lookups
  cost nothing on the request path; per-call overrides (`provider!/2`) build a fresh
  struct rather than writing to the cache. A module that is missing, lacks `new/1`, does
  not implement `ExAgent.Provider`, or whose `new/1` raises fails the boot with the role
  name in the message, so a missing credential crashes at deploy time instead of on the
  first request. Option values may be a zero-arity function or `{m, f, a}`, resolved once
  at boot, for vault-backed credentials.

  Note that this is the first thing in `lib/` to read `Application.get_env` — the library
  was otherwise configured entirely through explicit structs, and still can be.

- **`ExAgent.Error`.** A normalized error struct returned by every provider operation,
  carrying `:type`, `:message`, `:status`, `:provider`, `:raw` and `:retryable?`. HTTP
  statuses are classified into one vocabulary (`:auth`, `:rate_limit`,
  `:context_length`, `:invalid_request`, `:not_found`, `:timeout`, `:server`,
  `:transport`, `:unsupported`), so retry logic is written once rather than per
  provider. `ExAgent.Error.from_result/2` does the classification for custom providers.
  The struct is also an exception, so it can be raised where no return value exists.

- **URL file sources.** `files: [%{url: "https://..."}]` hands the URL straight to the
  provider — Gemini as `file_data.file_uri`, OpenAI as `image_url` / `file_url`. ExAgent
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
  property of the *model*, not the vendor — `o1-mini` reads no images. Defaults are
  `[:text, :image, :document]` for OpenAI, those plus `:video` and `:audio` for Gemini,
  and `[:text]` for `OpenAICompatible` (one container serves one model). Narrowing makes
  the gate fire locally instead of letting the provider 400 later:

      OpenAI.new(api_key: key, model: "o1-mini", modalities: [:text])

  ExAgent keeps no model-to-modality table on purpose: it would go stale silently, and
  whoever picked the model already knows.

- **Embeddings.** `ExAgent.embed(provider, inputs, opts)` returns an
  `%ExAgent.Embeddings{}` carrying `vectors`, `model`, `provider`, `dimensions`, `task`,
  and `usage`. It takes a provider struct rather than an agent pid — embedding is
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
  than guessing a family — pass `:embedding_family` to adopt a newer one.

  On `OpenAICompatible` only, `:task` also accepts a **raw string**, sent verbatim with
  no translation and no validation — a self-hosted endpoint serves whatever model you
  deployed, and those vocabularies change between versions (Jina v3's
  `"retrieval.passage"` became a `"retrieval"` task plus a prompt in v5, which also added
  `"text-matching"`). Gemini and OpenAI take atoms only: `taskType` is a closed enum and
  OpenAI has no task field, so a string there is a typo far more often than a new value
  and is rejected naming the valid atoms. Atoms stay validated everywhere, and the result
  carries back exactly what was passed.

  `ExAgent.Embeddings` also exposes `tasks/0`, `valid_task?/1`, `l2_normalize/1`, and
  `cosine_similarity/2`. **Persist `model`, `dimensions`, and `task` alongside every
  vector** — embedding spaces are model-scoped and mixing them degrades retrieval
  silently.
- **`ExAgent.Providers.OpenAICompatible`.** One provider for any endpoint speaking the
  OpenAI chat-completions dialect — self-hosted vLLM (including behind Modal),
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
  from ~10 seconds) — large files and video need it.
- **`ExAgent.UploadCache`.** An ETS-backed cache that lets the same bytes reuse an
  existing `ExAgent.FileRef` instead of re-uploading. Entries are keyed by
  `{scope, sha256(bytes)}` where the scope digests the provider module, base URL, and
  API key — so two accounts never share a file reference, and the key itself is never
  stored. An expired `FileRef` is treated as a miss and evicted. Added to the
  supervision tree ahead of the agent supervisor; `clear/0` empties it.

### Fixed

- **A failed turn poisoned the agent.** The user message was committed to context even
  when the turn failed, so a message the provider had already refused — a rejected
  attachment, say — was resent on every later turn and every one of them failed. "Fails
  loudly" became "fails forever". A failed turn now leaves no trace, which also stops a
  retry after a transient 429 from duplicating the question in history.

- **`chat_stream/3` raised on a rejected attachment.** The modality gate raises inside
  `ExAgent.Provider.stream/3` because a lazy enumerable has nowhere to carry an error at
  construction time, and that escaped to the consumer — contradicting the documented
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
  called `new/1`, so the closed list protected nothing — it only walled out everyone
  else. Any atom is now accepted, and a reference need only carry a `:file_id` or a
  `:file_uri`; OpenAI's and Gemini's specific field requirements still apply to them,
  since their services pattern-match on those fields.

- **`OpenAICompatible` shaped documents as images.** `:document` was declarable through
  `:modalities` but `format_attachment/1` fell through to `image_url`, so a PDF was sent
  as an image part and the gateway either rejected it or read nothing. Documents now use
  the dialect's `file` part — `file_data` for bytes (with the required `filename`) and
  `file_url` for a URL — which is what a gateway fronting a document-reading model
  expects. The moduledoc previously claimed documents were unsupported; a model behind
  OpenRouter or Modal may well read them, so it is a deployment property like every other
  modality.

- **Gemini streaming produced no text at all.** Gemini terminates SSE events with CRLF,
  but `ExAgent.SSE.take_events/1` split only on `"\n\n"`. A CRLF stream contains no such
  boundary, so every frame stayed buffered, no frame was ever decoded, and the stream
  ended with its terminal chunk and empty content — silently, with no error. Framing now
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
  only `{:ok, _}` / `{:error, _}` / `{:handoff, _, _}` — anything else raised
  `CaseClauseError` inside the supervised task, surfacing as an opaque `:server` error. An
  unwrapped return is now taken as the result.

- **Documented that OpenAI's `:web_search` needs `temperature: nil`.** `web_search_options`
  is only accepted by a `*-search-preview` model, and those reject `temperature` — so the
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
  remainder. A frame split across TCP reads is reassembled correctly — now covered by a
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

  Reasoning traces still arrive as `:thinking_delta` chunks — the `reasoning_content`
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
  must return a `Response` — build one with `ExAgent.Response.new/2`.
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
  (and the README claimed they were silently ignored — neither was right). Attaching a
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