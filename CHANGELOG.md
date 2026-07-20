## v0.2.0 (2026-07-20)

### Added

- **Streaming.** `ExAgent.chat_stream/3` (agent-level) and `ExAgent.Provider.stream/3`
  (provider-level) return a lazy `Stream` of text chunks. Tool-call turns are
  resolved non-streamed; only the final assistant turn is streamed. All three
  providers (OpenAI, Gemini, DeepSeek) implement the optional `stream/3` callback.
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