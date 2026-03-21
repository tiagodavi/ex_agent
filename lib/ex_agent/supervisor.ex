defmodule ExAgent.AgentSupervisor do
  @moduledoc """
  Top-level supervisor for the ExAgent OTP tree.

  Manages:
  - `ExAgent.AgentDynamicSupervisor` for runtime-spawned agents
  - `ExAgent.TaskSupervisor` for async LLM calls and parallel dispatch
  """

  use Supervisor

  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts \\ []) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = [
      {DynamicSupervisor, name: ExAgent.AgentDynamicSupervisor, strategy: :one_for_one},
      {Task.Supervisor, name: ExAgent.TaskSupervisor}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
