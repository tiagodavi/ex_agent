## v0.2.0 (2026-07-20)

### Changed

- **Providers are now a behaviour instead of protocols.** The `ExAgent.LlmProvider`
  and `ExAgent.FileUploader` protocols were removed and replaced by a single
  `ExAgent.Provider` behaviour (`chat/3` required, `upload/4` optional). Custom
  providers now declare `@behaviour ExAgent.Provider` and implement `chat/3`
  (and optionally `upload/4`) as public functions instead of using `defimpl`.

## v0.1.0 (2026-03-30)

First release!