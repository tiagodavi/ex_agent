defmodule ExAgent.Patterns.Handoff do
  @moduledoc """
  State-driven transitions pattern.

  Enables dynamic transfer of control between agent processes.
  A handoff tool is added to an agent's tools list. When the LLM
  invokes it, the agent's tool loop returns a `{:handoff, target, context}`
  tuple to the caller, who can then redirect future messages.
  """

  alias ExAgent.{Context, Message, Tool}

  @type handoff_result :: {:handoff, target :: pid() | atom(), Context.t()}

  @type spec :: %{
          required(:name) => String.t(),
          required(:agent) => pid() | atom(),
          required(:description) => String.t()
        }

  @doc """
  Builds one handoff tool per target agent.

  Each tool is named `handoff_to_<name>` and takes a `"summary"` argument the
  model fills in before transferring.

      Handoff.tools([
        %{name: "billing", agent: billing_agent, description: "Transfer billing questions"},
        %{name: "tech", agent: tech_agent, description: "Transfer technical problems"}
      ])
  """
  @spec tools([spec()]) :: [Tool.t()]
  def tools(specs) when is_list(specs), do: Enum.map(specs, &tool/1)

  @spec tool(spec()) :: Tool.t()
  defp tool(%{name: name, agent: target, description: description}) do
    %Tool{
      name: "handoff_to_#{name}",
      description: description,
      parameters: %{
        "type" => "object",
        "properties" => %{
          "summary" => %{
            "type" => "string",
            "description" => "Summary of the conversation context for the receiving agent"
          }
        },
        "required" => ["summary"]
      },
      function: fn args ->
        # Models skip "required" arguments; handing off with an empty summary
        # beats crashing the turn that asked for it.
        summary = Map.get(args, "summary") || "(no summary provided)"
        {:handoff, target, transfer_context(Context.new(), %{"summary" => summary})}
      end
    }
  end

  @doc """
  Produces a trimmed context suitable for the receiving agent.

  Adds a system message summarizing the handoff.
  """
  @spec transfer_context(Context.t(), map()) :: Context.t()
  def transfer_context(context, %{"summary" => summary}) do
    {:ok, msg} =
      Message.new(
        role: :user,
        content: "Handoff received. Previous context summary: #{summary}"
      )

    Context.add_message(context, msg)
  end

  @doc """
  Delivers a context to the target agent, which adopts it and returns to idle.

  This is the pattern's entry point, named `run/2` like every other pattern.
  Unlike the others it returns `:ok` rather than a result: the work here is the
  transfer itself.
  """
  @spec run(pid() | atom(), Context.t()) :: :ok
  def run(target, %Context{} = context) do
    GenServer.cast(target, {:receive_handoff, context})
  end
end
