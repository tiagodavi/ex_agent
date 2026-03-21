defmodule ExAgent.AgentDynamicSupervisor do
  @moduledoc """
  Dynamic supervisor for managing agent lifecycle at runtime.

  Provides functions to start and stop agent processes dynamically.
  """

  @doc """
  Starts a new agent under the dynamic supervisor.

  ## Options

  See `ExAgent.Agent.start_link/1` for available options.
  """
  @spec start_agent(ExAgent.Agent.agent_opts()) :: DynamicSupervisor.on_start_child()
  def start_agent(opts) do
    DynamicSupervisor.start_child(__MODULE__, {ExAgent.Agent, opts})
  end

  @doc """
  Stops an agent process.
  """
  @spec stop_agent(pid()) :: :ok | {:error, :not_found}
  def stop_agent(pid) do
    DynamicSupervisor.terminate_child(__MODULE__, pid)
  end
end
