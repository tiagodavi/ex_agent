defmodule ExAgent.Telemetry do
  @moduledoc """
  `:telemetry` events emitted around every provider call.

  A library whose job is calling billed APIs has to be measurable: latency,
  token spend, failure rate, and tool-loop depth are operational facts, not
  debugging trivia. Attaching a handler is the supported way to get them —
  nothing here logs on your behalf.

  ## Events

  | Event | Measurements | Metadata |
  |-------|--------------|----------|
  | `[:ex_agent, :chat, :start]` | `:system_time` | `:provider`, `:model`, `:message_count`, `:streaming?` |
  | `[:ex_agent, :chat, :stop]` | `:duration`, plus `:input_tokens`, `:output_tokens`, `:total_tokens` when reported | `:provider`, `:model`, `:result`, `:streaming?` |
  | `[:ex_agent, :chat, :exception]` | `:duration` | `:provider`, `:model`, `:kind`, `:reason`, `:stacktrace` |
  | `[:ex_agent, :embed, :start]` | `:system_time` | `:provider`, `:input_count` |
  | `[:ex_agent, :embed, :stop]` | `:duration`, `:input_tokens`, `:total_tokens` | `:provider`, `:model`, `:result` |
  | `[:ex_agent, :tool, :stop]` | `:duration` | `:tool`, `:result` |

  `:result` is `:ok`, `:tool_calls`, or `:error`. On `:error` the metadata also
  carries `:error_type` and `:retryable?` from `ExAgent.Error`, which is what a
  retry-rate dashboard is actually built on.

  ## Example

      :telemetry.attach(
        "ex-agent-cost",
        [:ex_agent, :chat, :stop],
        fn _event, measurements, metadata, _config ->
          MyApp.Metrics.record(metadata.model, measurements[:total_tokens] || 0)
        end,
        nil
      )

  `:telemetry` is a hard dependency but a tiny one, and emitting with no handler
  attached costs a single ETS lookup.
  """

  @doc """
  Runs `fun`, emitting `:start`/`:stop`/`:exception` events under `[:ex_agent | name]`.
  """
  @spec span([atom()], map(), (-> result)) :: result when result: term()
  def span(name, metadata, fun) do
    :telemetry.span([:ex_agent | name], metadata, fn ->
      result = fun.()
      {result, measurements(result), Map.merge(metadata, result_metadata(result))}
    end)
  end

  # `:telemetry.span/3` wants the extra measurements separately from the value,
  # so usage counts are lifted out of the result here rather than at each call site.
  @spec measurements(term()) :: map()
  defp measurements({:ok, %{usage: usage}}) when is_map(usage) do
    usage
    |> Map.take([:input_tokens, :output_tokens, :total_tokens])
    |> Map.reject(fn {_key, value} -> is_nil(value) end)
  end

  defp measurements(_result), do: %{}

  @spec result_metadata(term()) :: map()
  defp result_metadata({:ok, %{model: model}}), do: %{result: :ok, model: model}
  defp result_metadata({:ok, _value}), do: %{result: :ok}

  defp result_metadata({:tool_calls, calls}),
    do: %{result: :tool_calls, tool_count: length(calls)}

  defp result_metadata({:error, %ExAgent.Error{} = error}),
    do: %{result: :error, error_type: error.type, retryable?: error.retryable?}

  defp result_metadata({:error, _reason}), do: %{result: :error}
  defp result_metadata(_result), do: %{result: :ok}
end
