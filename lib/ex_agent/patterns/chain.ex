defmodule ExAgent.Patterns.Chain do
  @moduledoc """
  Sequential steps, each working on what the last one produced.

  **The analogy: an assembly line.** Each station does one small job to the thing
  in front of it and passes it along. Nobody decides where the work goes — the
  order is fixed by you, in code.

  Use it when you already know the steps. "Transcribe, then extract the action
  items, then rewrite them as a checklist" is three prompts in a fixed order, and
  a single prompt asking for all three at once does each of them worse.

      alias ExAgent.Patterns.Chain

      {:ok, checklist} =
        Chain.run(transcript,
          steps: [
            Chain.llm(provider, &"Extract the action items:\\n\\n\#{&1}"),
            Chain.llm(provider, &"Rewrite these as a markdown checklist:\\n\\n\#{&1}")
          ]
        )

  ## Gates between steps

  A step is any function from the previous value to `{:ok, next}`,
  `{:error, reason}`, or `{:halt, value}`. `:halt` stops the line early and is
  *not* a failure — it is how you decline to spend the rest of the calls:

      Chain.run(ticket,
        steps: [
          Chain.llm(triage, &"Is this a bug report? Answer YES or NO.\\n\\n\#{&1}"),
          fn answer ->
            if String.starts_with?(answer, "YES"), do: {:ok, ticket}, else: {:halt, :not_a_bug}
          end,
          Chain.llm(engineer, &"Suggest a fix:\\n\\n\#{&1}")
        ]
      )

  A gate is also where a human belongs. Return `{:halt, :needs_approval}`, park
  the work, and start a second chain when someone approves it — the pattern needs
  nothing special for that.

  ## When not to use it

  If the *order* depends on the input, you want `ExAgent.Patterns.Router`. If the
  steps are unknown until an LLM decides them, you want
  `ExAgent.Patterns.Subagents`. A chain is deliberately dumb: that is what makes
  it cheap to debug.
  """

  alias ExAgent.Patterns.Ask

  @typedoc """
  Where to send a prompt: a provider struct for a stateless call, or a running
  agent when the step should remember the conversation.
  """
  @type target :: struct() | GenServer.server()

  @type value :: term()
  @type step :: (value() -> {:ok, value()} | {:error, term()} | {:halt, value()})

  @type chain_opts :: [steps: [step()]]

  @doc """
  Runs `input` through every step in order.

  Returns `{:ok, value}` when every step succeeded, `{:halted, value}` when a step
  stopped the line, and `{:error, reason}` on the first failure — along with the
  index of the step that failed, since "step 3 of 5" is the first thing you want
  to know.
  """
  @spec run(value(), chain_opts()) ::
          {:ok, value()} | {:halted, value()} | {:error, {non_neg_integer(), term()}}
  def run(input, opts) do
    steps = Keyword.fetch!(opts, :steps)

    steps
    |> Enum.with_index()
    |> Enum.reduce_while({:ok, input}, fn {step, index}, {:ok, value} ->
      case step.(value) do
        {:ok, next} -> {:cont, {:ok, next}}
        {:halt, halted} -> {:halt, {:halted, halted}}
        {:error, reason} -> {:halt, {:error, {index, reason}}}
        other -> {:halt, {:error, {index, {:invalid_step_result, other}}}}
      end
    end)
  end

  @doc """
  Builds a step that sends a prompt to `target` and returns its text.

  `target` is a provider struct for a stateless call, or a running agent when the
  step should remember the conversation. `build_prompt` turns the previous value
  into the prompt.

      Chain.llm(provider, fn summary -> "Translate to French:\\n\\n\#{summary}" end)
  """
  @spec llm(target(), (value() -> String.t())) :: step()
  def llm(target, build_prompt) when is_function(build_prompt, 1) do
    fn value -> Ask.ask(target, build_prompt.(value)) end
  end
end
