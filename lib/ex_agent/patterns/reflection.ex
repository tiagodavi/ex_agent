defmodule ExAgent.Patterns.Reflection do
  @moduledoc """
  Draft, critique, revise - until a reviewer signs off or the budget runs out.

  **The analogy: a writer and an editor.** The writer produces a draft. The editor
  marks it up. The writer revises. Two or three rounds of that beats one person
  trying to be brilliant on the first attempt.

  Use it when quality is the bottleneck and you can *say* what good looks like:
  code that has to compile, a summary that must not exceed 200 words, SQL that
  must only read from three tables. Reflection is the pattern that turns "usually
  fine" into "checked".

      alias ExAgent.Patterns.Reflection

      {:ok, result} =
        Reflection.run("Write a Postgres query for monthly active users.",
          generator: coder,
          critic: reviewer,
          accept?: &String.contains?(&1, "APPROVED"),
          max_rounds: 3
        )

      result.output   # the accepted draft
      result.rounds   # how many revisions it took

  ## The critic needs a way to say yes

  `accept?` decides when to stop. Ask the critic for a token you can match on and
  keep the instruction blunt - "Reply APPROVED if it is correct, otherwise list
  what is wrong". Without a stop condition the loop just burns `max_rounds` worth
  of tokens every time.

  ## Two guardrails, both deliberate

  `max_rounds` defaults to `3` and is a hard ceiling: an LLM critic can always
  find something to complain about, so an unbounded loop is a runaway bill.

  When the ceiling is hit without approval you get `{:max_rounds, result}`, not
  `{:ok, result}`. The last draft is still there - often it is good enough - but
  you have to *choose* to use unapproved work rather than have it handed to you as
  if a reviewer had passed it.

  ## When not to use it

  If you cannot describe the acceptance criteria, a critic will produce vague
  praise and you will pay double for the same answer. If the risk is a wrong
  *fact* rather than poor quality, prefer `ExAgent.Patterns.Consensus` - a critic
  reading one draft is easy to talk into agreeing with it.
  """

  alias ExAgent.Patterns.Ask

  @typedoc """
  Where to send a prompt: a provider struct for a stateless call, or a running
  agent when the step should remember the conversation.
  """
  @type target :: struct() | GenServer.server()

  @default_max_rounds 3

  @type result :: %{
          output: String.t(),
          rounds: non_neg_integer(),
          critiques: [String.t()],
          accepted?: boolean()
        }

  @type reflection_opts :: [
          generator: target(),
          critic: target(),
          accept?: (String.t() -> boolean()),
          max_rounds: pos_integer(),
          critic_prompt: (String.t(), String.t() -> String.t()),
          revise_prompt: (String.t(), String.t(), String.t() -> String.t())
        ]

  @doc """
  Generates an answer to `task`, then critiques and revises it.

  ## Options

  - `:generator` (required) - provider or agent that drafts and revises
  - `:critic` - provider or agent that reviews (default: the generator, which is
    cheaper but a weaker check - a model reviewing itself agrees with itself)
  - `:accept?` - `(critique -> boolean)` deciding when the draft is good
    (default: the critique contains `"APPROVED"`)
  - `:max_rounds` - hard ceiling on revisions (default: `#{@default_max_rounds}`)
  - `:critic_prompt` - `(task, draft -> prompt)` to override the review prompt
  - `:revise_prompt` - `(task, draft, critique -> prompt)` to override the revision

  Returns `{:ok, result}` once the critic accepts, `{:max_rounds, result}` if it
  never does, or `{:error, reason}` if a call fails.
  """
  @spec run(String.t(), reflection_opts()) ::
          {:ok, result()} | {:max_rounds, result()} | {:error, term()}
  def run(task, opts) when is_binary(task) do
    generator = Keyword.fetch!(opts, :generator)
    critic = Keyword.get(opts, :critic, generator)
    accept? = Keyword.get(opts, :accept?, &default_accept?/1)
    max_rounds = Keyword.get(opts, :max_rounds, @default_max_rounds)

    with {:ok, draft} <- Ask.ask(generator, task) do
      refine(task, draft, [], 0, %{
        generator: generator,
        critic: critic,
        accept?: accept?,
        max_rounds: max_rounds,
        critic_prompt: Keyword.get(opts, :critic_prompt, &default_critic_prompt/2),
        revise_prompt: Keyword.get(opts, :revise_prompt, &default_revise_prompt/3)
      })
    end
  end

  @spec refine(String.t(), String.t(), [String.t()], non_neg_integer(), map()) ::
          {:ok, result()} | {:max_rounds, result()} | {:error, term()}
  defp refine(_task, draft, critiques, rounds, %{max_rounds: max}) when rounds >= max do
    {:max_rounds, result(draft, rounds, critiques, false)}
  end

  defp refine(task, draft, critiques, rounds, config) do
    with {:ok, critique} <- Ask.ask(config.critic, config.critic_prompt.(task, draft)) do
      critiques = critiques ++ [critique]

      if config.accept?.(critique) do
        {:ok, result(draft, rounds, critiques, true)}
      else
        prompt = config.revise_prompt.(task, draft, critique)

        with {:ok, revised} <- Ask.ask(config.generator, prompt) do
          refine(task, revised, critiques, rounds + 1, config)
        end
      end
    end
  end

  @spec result(String.t(), non_neg_integer(), [String.t()], boolean()) :: result()
  defp result(output, rounds, critiques, accepted?) do
    %{output: output, rounds: rounds, critiques: critiques, accepted?: accepted?}
  end

  @spec default_accept?(String.t()) :: boolean()
  defp default_accept?(critique), do: String.contains?(critique, "APPROVED")

  @spec default_critic_prompt(String.t(), String.t()) :: String.t()
  defp default_critic_prompt(task, draft) do
    """
    Task: #{task}

    Draft:
    #{draft}

    Review the draft against the task. If it fully satisfies the task, reply with
    exactly APPROVED. Otherwise list the specific problems, one per line, with no
    preamble.
    """
  end

  @spec default_revise_prompt(String.t(), String.t(), String.t()) :: String.t()
  defp default_revise_prompt(task, draft, critique) do
    """
    Task: #{task}

    Your previous attempt:
    #{draft}

    A reviewer raised these problems:
    #{critique}

    Rewrite the answer so every problem is addressed. Output only the revised
    answer, with no commentary about the changes.
    """
  end
end
