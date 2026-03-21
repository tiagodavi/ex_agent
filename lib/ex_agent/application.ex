defmodule ExAgent.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      ExAgent.AgentSupervisor
    ]

    opts = [strategy: :one_for_one, name: ExAgent.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
