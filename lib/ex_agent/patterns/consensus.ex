defmodule ExAgent.Patterns.Consensus do
  @moduledoc """
  Ask the same question several times, then go with the answer that keeps coming up.

  **The analogy: a second opinion.** You do not ask one doctor to explain their
  reasoning more carefully; you ask three doctors and see whether they agree.

  Use it when being *wrong* is expensive and the answer is short enough to compare:
  a classification, a routing decision, an extracted number, a yes/no. Independent
  attempts catch the case where a model is confidently mistaken, which no amount of
  self-review will - a critic reading one draft is easily talked into agreeing
  with it.

      alias ExAgent.Patterns.Consensus

      {:ok, verdict} =
        Consensus.run("Is this email a phishing attempt? Answer YES or NO.\\n\\n" <> body,
          voters: provider,
          samples: 5
        )

      verdict.answer      # "YES"
      verdict.agreement   # 0.8  - four of five agreed
      verdict.votes       # %{"YES" => 4, "NO" => 1}

  ## Disagreement is the useful part

  The point is not only the winning answer but how lopsided the vote was.
  `:agreement` is the winner's share, and a low number is your signal to escalate
  - to a bigger model, to a human, or to declining to answer:

      case Consensus.run(question, voters: provider, samples: 5) do
        {:ok, %{agreement: a} = verdict} when a >= 0.8 -> {:auto, verdict.answer}
        {:ok, verdict} -> {:needs_review, verdict}
      end

  ## Different voters beat repeated voters

  Passing one provider samples it repeatedly, which only helps if its output
  varies - at `temperature: 0` you will pay five times for one answer. Passing a
  *list* of providers is stronger, because different models fail differently:

      Consensus.run(question, voters: [gpt, gemini, local_llama])

  ## Normalize before counting

  Votes are tallied on exact strings after trimming and downcasing, so `"YES"` and
  `"Yes."` are different answers unless you say otherwise. For anything but a
  strict format, pass `:normalize`:

      Consensus.run(question,
        voters: provider,
        samples: 5,
        normalize: fn answer -> answer |> String.upcase() |> String.contains?("YES") end
      )

  ## When not to use it

  Long prose does not vote - two summaries are never byte-identical, so every
  answer wins with `1/n`. Use `ExAgent.Patterns.Reflection` to improve a long
  answer, and consensus to decide a short one.
  """

  alias ExAgent.Patterns.Ask

  @typedoc """
  Where to send a prompt: a provider struct for a stateless call, or a running
  agent when the step should remember the conversation.
  """
  @type target :: struct() | GenServer.server()

  @default_samples 3
  @default_timeout :timer.minutes(5)

  @type verdict :: %{
          answer: term(),
          agreement: float(),
          votes: %{optional(term()) => pos_integer()},
          answers: [String.t()],
          failures: [term()]
        }

  @type consensus_opts :: [
          voters: target() | [target()],
          samples: pos_integer(),
          normalize: (String.t() -> term()),
          timeout: pos_integer()
        ]

  @doc """
  Puts `prompt` to several voters and returns the most common answer.

  ## Options

  - `:voters` (required) - one provider/agent, sampled `:samples` times, or a list
    of them, each asked once
  - `:samples` - how many times to ask a single voter (default:
    `#{@default_samples}`); ignored when `:voters` is a list
  - `:normalize` - `(answer -> comparable)` applied before votes are counted
    (default: trim and downcase)
  - `:timeout` - per voter, in milliseconds (default: 5 minutes)

  Returns `{:ok, verdict}`, or `{:error, :no_answers}` if every voter failed. Ties
  are broken by first appearance, so the winner is always the earliest of the
  joint-highest - deterministic, but check `:agreement` before trusting a tie.
  """
  @spec run(String.t(), consensus_opts()) :: {:ok, verdict()} | {:error, term()}
  def run(prompt, opts) when is_binary(prompt) do
    targets = voters(opts)
    normalize = Keyword.get(opts, :normalize, &default_normalize/1)

    {answers, failures} = collect(prompt, targets, Keyword.get(opts, :timeout, @default_timeout))

    case answers do
      [] -> {:error, :no_answers}
      answers -> {:ok, tally(answers, failures, normalize)}
    end
  end

  @spec voters(keyword()) :: [target()]
  defp voters(opts) do
    case Keyword.fetch!(opts, :voters) do
      targets when is_list(targets) -> targets
      target -> List.duplicate(target, Keyword.get(opts, :samples, @default_samples))
    end
  end

  @spec collect(String.t(), [target()], pos_integer()) :: {[String.t()], [term()]}
  defp collect(prompt, targets, timeout) do
    targets
    |> Task.async_stream(&Ask.ask(&1, prompt),
      timeout: timeout,
      on_timeout: :kill_task,
      zip_input_on_exit: true
    )
    |> Enum.reduce({[], []}, fn
      {:ok, {:ok, answer}}, {answers, failures} -> {answers ++ [answer], failures}
      {:ok, {:error, reason}}, {answers, failures} -> {answers, failures ++ [reason]}
      {:exit, {_target, reason}}, {answers, failures} -> {answers, failures ++ [reason]}
    end)
  end

  @spec tally([String.t()], [term()], (String.t() -> term())) :: verdict()
  defp tally(answers, failures, normalize) do
    normalized = Enum.map(answers, normalize)
    votes = Enum.frequencies(normalized)

    # Candidates are ranked in first-appearance order, not map order: iterating
    # `votes` directly would break ties arbitrarily, because a map has no
    # insertion order to fall back on. `Enum.uniq/1` preserves call order and
    # `max_by/2` keeps the first of equal counts, so a tie is deterministic.
    winner = normalized |> Enum.uniq() |> Enum.max_by(&Map.fetch!(votes, &1))
    count = Map.fetch!(votes, winner)

    %{
      answer: winner,
      agreement: count / length(normalized),
      votes: votes,
      answers: answers,
      failures: failures
    }
  end

  @spec default_normalize(String.t()) :: String.t()
  defp default_normalize(answer), do: answer |> String.trim() |> String.downcase()
end
