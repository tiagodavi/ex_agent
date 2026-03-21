defmodule ExAgent.Patterns.Router do
  @moduledoc """
  Parallel dispatch and synthesis pattern.

  A stateless module that classifies input, dispatches to multiple
  specialized agents in parallel, and synthesizes results into a
  single response.
  """

  alias ExAgent.{Agent, Message}

  @type route :: %{
          name: String.t(),
          agent: pid() | atom(),
          match_fn: (String.t() -> boolean())
        }

  @type router_opts :: [
          routes: [route()],
          synthesizer: (String.t(), [{String.t(), String.t()}] -> String.t()),
          timeout: pos_integer()
        ]

  @doc """
  Routes input through matching agents and synthesizes results.

  1. Classifies the input by evaluating each route's `match_fn`
  2. Dispatches to all matching agents in parallel
  3. Synthesizes the results using the provided synthesizer function

  ## Options

  - `:routes` (required) - list of route maps with `:name`, `:agent`, `:match_fn`
  - `:synthesizer` - function `(input, results) -> combined_string` (default: joins with headers)
  - `:timeout` - milliseconds to wait for each agent (default: 30_000)
  """
  @spec route(String.t(), router_opts()) :: {:ok, String.t()} | {:error, term()}
  def route(input, opts) do
    routes = Keyword.fetch!(opts, :routes)
    synthesizer = Keyword.get(opts, :synthesizer, &default_synthesizer/2)
    timeout = Keyword.get(opts, :timeout, 30_000)

    matched_routes = classify(input, routes)

    case matched_routes do
      [] ->
        {:error, :no_matching_routes}

      routes ->
        results = dispatch(input, routes, timeout)
        {:ok, synthesizer.(input, results)}
    end
  end

  @doc """
  Evaluates each route's match_fn against the input.

  Returns all routes whose `match_fn` returns `true`.
  """
  @spec classify(String.t(), [route()]) :: [route()]
  def classify(input, routes) do
    Enum.filter(routes, fn %{match_fn: match_fn} -> match_fn.(input) end)
  end

  @doc """
  Dispatches input to all given routes' agents in parallel.

  Returns a list of `{route_name, response_content}` tuples.
  Failed agents are included with an error message.
  """
  @spec dispatch(String.t(), [route()], pos_integer()) :: [{String.t(), String.t()}]
  def dispatch(input, routes, timeout \\ 30_000) do
    routes
    |> Task.async_stream(
      fn %{name: name, agent: agent} ->
        case Agent.chat(agent, input) do
          {:ok, %Message{content: content}} -> {name, content}
          {:error, reason} -> {name, "Error: #{inspect(reason)}"}
        end
      end,
      timeout: timeout,
      on_timeout: :kill_task
    )
    |> Enum.map(fn
      {:ok, result} -> result
      {:exit, :timeout} -> {"unknown", "Error: timeout"}
    end)
  end

  @doc """
  Combines results with route name headers.
  """
  @spec synthesize(String.t(), [{String.t(), String.t()}], fun()) :: String.t()
  def synthesize(input, results, synthesizer \\ &default_synthesizer/2) do
    synthesizer.(input, results)
  end

  @spec default_synthesizer(String.t(), [{String.t(), String.t()}]) :: String.t()
  defp default_synthesizer(_input, results) do
    results
    |> Enum.map(fn {name, content} -> "## #{name}\n#{content}" end)
    |> Enum.join("\n\n")
  end
end
