defmodule ExAgent.Patterns.MapReduce do
  @moduledoc """
  Split one big input into pieces, process them in parallel, combine the results.

  **The analogy: reading a long report with colleagues.** You each take a chapter,
  summarise it, then someone stitches the summaries into one. Nobody reads the
  whole thing.

  Use it when the input does not fit - or fits but answers worse whole than in
  parts. A 300-page contract, a week of logs, forty customer interviews.

      alias ExAgent.Patterns.MapReduce

      {:ok, summary} =
        MapReduce.run(chapters, provider,
          map: &"Summarise this chapter in three bullets:\\n\\n\#{&1}",
          reduce: fn summaries ->
            {:ok, "Overall:\\n" <> Enum.join(summaries, "\\n")}
          end
        )

  Pass `reduce: {provider, prompt_builder}` to have a model do the combining,
  which is the usual case:

      MapReduce.run(chapters, worker,
        map: &"Summarise:\\n\\n\#{&1}",
        reduce: {editor, fn parts -> "Merge these into one summary:\\n\\n" <> Enum.join(parts, "\\n---\\n") end}
      )

  ## Sections are independent, and that is the catch

  Each piece is processed with no knowledge of the others, so anything that spans a
  boundary is invisible: a clause on page 40 that contradicts page 7, a log line
  that only matters next to one an hour earlier. Overlap your sections, or make the
  reduce step look for conflicts explicitly.

  ## Partial failure is reported, not hidden

  One section failing does not fail the run. The reduce step receives only the
  sections that succeeded, and `:failures` on the result tells you what was
  missing - a summary built from 38 of 40 interviews is usually still worth having,
  but never worth mistaking for all 40.

  ## When not to use it

  If the pieces need to know about each other, this is the wrong shape - chain them
  with `ExAgent.Patterns.Chain` so each sees the last. If you are sending the
  *same* input to several models rather than different inputs to one, you want
  `ExAgent.Patterns.Consensus`.
  """

  alias ExAgent.Patterns.Ask

  @typedoc """
  Where to send a prompt: a provider struct for a stateless call, or a running
  agent when the step should remember the conversation.
  """
  @type target :: struct() | GenServer.server()

  @default_timeout :timer.minutes(5)

  @type section :: term()

  @type result :: %{
          output: term(),
          sections: non_neg_integer(),
          failures: [{non_neg_integer(), term()}]
        }

  @type reduce ::
          ([String.t()] -> {:ok, term()} | {:error, term()})
          | {target(), ([String.t()] -> String.t())}

  @type map_reduce_opts :: [
          map: (section() -> String.t()),
          reduce: reduce(),
          timeout: pos_integer(),
          max_concurrency: pos_integer()
        ]

  @doc """
  Maps `sections` through `target` in parallel, then reduces the outputs.

  ## Options

  - `:map` (required) - `(section -> prompt)`
  - `:reduce` (required) - either `(outputs -> {:ok, value} | {:error, reason})`, or
    `{target, (outputs -> prompt)}` to have a model combine them
  - `:timeout` - per section, in milliseconds (default: 5 minutes)
  - `:max_concurrency` - parallel sections (default: `System.schedulers_online/0`,
    which is a poor proxy for how many requests a provider will tolerate - lower
    it if you are being rate limited)

  Returns `{:ok, result}`, or `{:error, {:all_sections_failed, failures}}` when
  nothing survived to reduce - the failures come with it, since "everything failed"
  on its own does not say why.
  """
  @spec run([section()], target(), map_reduce_opts()) ::
          {:ok, result()} | {:error, term()}
  def run(sections, target, opts) when is_list(sections) do
    build_prompt = Keyword.fetch!(opts, :map)
    reduce = Keyword.fetch!(opts, :reduce)

    {outputs, failures} = map_sections(sections, target, build_prompt, opts)

    if outputs == [] and sections != [] do
      # Carrying the failures is the difference between a debuggable error and
      # "all sections failed", which says nothing about why.
      {:error, {:all_sections_failed, failures}}
    else
      with {:ok, output} <- apply_reduce(reduce, outputs) do
        {:ok, %{output: output, sections: length(outputs), failures: failures}}
      end
    end
  end

  @spec map_sections([section()], target(), (section() -> String.t()), keyword()) ::
          {[String.t()], [{non_neg_integer(), term()}]}
  defp map_sections(sections, target, build_prompt, opts) do
    sections
    |> Enum.with_index()
    |> Task.async_stream(
      fn {section, index} ->
        {index, Ask.ask(target, build_prompt.(section))}
      end,
      timeout: Keyword.get(opts, :timeout, @default_timeout),
      max_concurrency: Keyword.get(opts, :max_concurrency, System.schedulers_online()),
      on_timeout: :kill_task,
      # Without this a timeout carries no section number, and "one of them timed
      # out" is not a debuggable statement.
      zip_input_on_exit: true
    )
    # Prepend and reverse once: appending with ++ per section is quadratic, and a
    # sectioned input is exactly the case where the list is long.
    |> Enum.reduce({[], []}, fn
      {:ok, {_index, {:ok, output}}}, {outputs, failures} ->
        {[output | outputs], failures}

      {:ok, {index, {:error, reason}}}, {outputs, failures} ->
        {outputs, [{index, reason} | failures]}

      {:exit, {{_section, index}, reason}}, {outputs, failures} ->
        {outputs, [{index, reason} | failures]}
    end)
    |> then(fn {outputs, failures} -> {Enum.reverse(outputs), Enum.reverse(failures)} end)
  end

  @spec apply_reduce(reduce(), [String.t()]) :: {:ok, term()} | {:error, term()}
  defp apply_reduce({target, build_prompt}, outputs) when is_function(build_prompt, 1) do
    Ask.ask(target, build_prompt.(outputs))
  end

  defp apply_reduce(reduce, outputs) when is_function(reduce, 1) do
    case reduce.(outputs) do
      {:ok, value} -> {:ok, value}
      {:error, reason} -> {:error, reason}
      other -> {:error, {:invalid_reduce_result, other}}
    end
  end
end
