defmodule ExAgent.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    # Before the tree starts: a bad role must crash at deploy time, not on the
    # first request.
    ExAgent.Roles.build!()

    children = [
      ExAgent.AgentSupervisor
    ]

    opts = [strategy: :one_for_one, name: ExAgent.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
