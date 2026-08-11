defmodule ExAgent.Patterns.Ask do
  @moduledoc false

  # One-shot "send this prompt, get text back", accepting either a provider
  # struct or a running agent.
  #
  # The patterns are otherwise forced to pick: `Router` took agents, `Subagents`
  # took provider structs, and neither could be handed the other. A stateless
  # step usually wants a bare provider (no process, no history); a step that
  # should remember the conversation wants an agent. Both are legitimate, so the
  # patterns accept either and this decides which call to make.

  alias ExAgent.{Message, Provider, Response}

  @type target :: struct() | GenServer.server()

  @doc false
  @spec ask(target(), String.t()) :: {:ok, String.t()} | {:error, term()}
  def ask(%_struct{} = provider, prompt) do
    with {:ok, message} <- build_message(prompt) do
      provider |> Provider.chat([message]) |> normalize()
    end
  end

  def ask(agent, prompt), do: agent |> ExAgent.Agent.chat(prompt) |> normalize()

  @spec build_message(String.t()) :: {:ok, Message.t()} | {:error, term()}
  defp build_message(prompt) do
    case Message.new(role: :user, content: prompt) do
      {:ok, message} -> {:ok, message}
      {:error, reason} -> {:error, ExAgent.Error.new(:invalid_request, reason)}
    end
  end

  @spec normalize(term()) :: {:ok, String.t()} | {:error, term()}
  defp normalize({:ok, %Response{content: content}}), do: {:ok, content}
  defp normalize({:error, reason}), do: {:error, reason}

  # A tool call or handoff is a legitimate answer to a chat request but not to a
  # pipeline step, which needs text to pass on.
  defp normalize({:tool_calls, calls}),
    do: {:error, {:unexpected_tool_calls, Enum.map(calls, & &1["name"])}}

  defp normalize({:handoff, target, _context}), do: {:error, {:unexpected_handoff, target}}
  defp normalize(other), do: {:error, {:unexpected_reply, other}}
end
