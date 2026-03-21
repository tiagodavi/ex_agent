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

  @doc """
  Creates a tool that triggers a handoff to the target agent when invoked.

  The tool accepts a `"summary"` parameter that the LLM uses to
  summarize the conversation before handoff.
  """
  @spec build_handoff_tool(String.t(), pid() | atom(), String.t()) :: Tool.t()
  def build_handoff_tool(name, target, description) do
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
      function: fn %{"summary" => summary} ->
        context = transfer_context(Context.new(), %{"summary" => summary})
        {:handoff, target, context}
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
  Sends a context to the target agent via GenServer.cast.
  """
  @spec execute_handoff(pid() | atom(), Context.t()) :: :ok
  def execute_handoff(target, %Context{} = context) do
    GenServer.cast(target, {:receive_handoff, context})
  end
end
