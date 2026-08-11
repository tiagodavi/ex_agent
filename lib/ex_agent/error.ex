defmodule ExAgent.Error do
  @moduledoc """
  Normalized error returned by every provider operation.

  Providers speak different error dialects; this struct gives callers one
  vocabulary. The `:type` classifies *what* went wrong, `:retryable?` says
  whether retrying could plausibly succeed, and `:raw` always preserves the
  original body so nothing is lost in translation.

      case ExAgent.chat(agent, "hello") do
        {:ok, message} -> message
        {:error, %ExAgent.Error{retryable?: true}} -> retry_with_backoff()
        {:error, %ExAgent.Error{} = error} -> Logger.error(error.message)
      end

  This is also an exception, so it can be raised where a return value is not
  available (see `ExAgent.Provider.stream/3`).

  ## Types

  | Type | Meaning | Retryable |
  |------|---------|-----------|
  | `:auth` | Bad or missing credentials (401, 403) | no |
  | `:not_found` | Unknown model or resource (404) | no |
  | `:timeout` | Request timed out (408, transport timeout) | yes |
  | `:rate_limit` | Rate or quota limit hit (429) | yes |
  | `:context_length` | Input exceeds the model's context window | no |
  | `:invalid_request` | Malformed request (other 4xx) | no |
  | `:unsupported` | The provider cannot perform this operation | no |
  | `:server` | Provider-side failure (5xx, unexpected response shape) | yes for 5xx |
  | `:transport` | Connection-level failure | yes |
  """

  @type type ::
          :auth
          | :rate_limit
          | :invalid_request
          | :context_length
          | :unsupported
          | :not_found
          | :server
          | :timeout
          | :transport

  @type t :: %__MODULE__{
          type: type(),
          message: String.t(),
          status: pos_integer() | nil,
          provider: module() | nil,
          raw: term(),
          retryable?: boolean()
        }

  defexception [:type, :message, :status, :provider, :raw, retryable?: false]

  @context_length_pattern ~r/context length|context window|maximum context|too many tokens|input is too long/i

  @doc """
  Builds an error that did not originate from an HTTP response.

  Used for capability errors (`:unsupported`) and local validation failures.
  """
  @spec new(type(), String.t(), module() | nil) :: t()
  def new(type, message, provider \\ nil) do
    %__MODULE__{
      type: type,
      message: message,
      provider: provider,
      retryable?: retryable?(type)
    }
  end

  @doc """
  Builds an error for a success response whose shape the service cannot parse.

  Retryable - a malformed body is usually a transient provider-side glitch.
  """
  @spec unexpected_response(term(), module() | nil) :: t()
  def unexpected_response(body, provider \\ nil) do
    %__MODULE__{
      type: :server,
      message: "unexpected response shape from #{inspect(provider)}",
      provider: provider,
      raw: body,
      retryable?: true
    }
  end

  @doc """
  Classifies a non-success HTTP response into a normalized error.

  `body` is preserved verbatim in `:raw`.
  """
  @spec from_http(pos_integer(), term(), module() | nil) :: t()
  def from_http(status, body, provider \\ nil) do
    type = classify(status, body)

    %__MODULE__{
      type: type,
      message: extract_message(body) || "HTTP #{status}",
      status: status,
      provider: provider,
      raw: body,
      retryable?: retryable?(type)
    }
  end

  @doc """
  Classifies a connection-level failure into a normalized error.
  """
  @spec from_transport(term(), module() | nil) :: t()
  def from_transport(reason, provider \\ nil) do
    type = if transport_timeout?(reason), do: :timeout, else: :transport

    %__MODULE__{
      type: type,
      message: transport_message(reason),
      provider: provider,
      raw: reason,
      retryable?: true
    }
  end

  @doc """
  Normalizes a `Req` result into `{:ok, body}` or `{:error, t()}`.

  Collapses the status/transport dispatch every service would otherwise repeat.
  """
  @spec from_result({:ok, Req.Response.t()} | {:error, term()}, module() | nil) ::
          {:ok, term()} | {:error, t()}
  def from_result(result, provider \\ nil)

  def from_result({:ok, %Req.Response{status: status, body: body}}, _provider)
      when status in 200..299,
      do: {:ok, body}

  def from_result({:ok, %Req.Response{status: status, body: body}}, provider),
    do: {:error, from_http(status, body, provider)}

  def from_result({:error, reason}, provider), do: {:error, from_transport(reason, provider)}

  @impl true
  def message(%__MODULE__{message: message}), do: message

  @spec classify(pos_integer(), term()) :: type()
  defp classify(status, _body) when status in [401, 403], do: :auth
  defp classify(404, _body), do: :not_found
  defp classify(408, _body), do: :timeout
  defp classify(429, _body), do: :rate_limit
  defp classify(status, _body) when status >= 500, do: :server

  defp classify(400, body) do
    if context_length?(body), do: :context_length, else: :invalid_request
  end

  defp classify(_status, _body), do: :invalid_request

  # Providers that expose a structured code are authoritative; the message
  # regex is the fallback for those that don't.
  @spec context_length?(term()) :: boolean()
  defp context_length?(%{"error" => %{"code" => "context_length_exceeded"}}), do: true

  defp context_length?(body) do
    case extract_message(body) do
      nil -> false
      message -> Regex.match?(@context_length_pattern, message)
    end
  end

  @spec extract_message(term()) :: String.t() | nil
  defp extract_message(%{"error" => %{"message" => message}}) when is_binary(message), do: message
  defp extract_message(%{"error" => message}) when is_binary(message), do: message
  defp extract_message(%{"message" => message}) when is_binary(message), do: message
  defp extract_message(body) when is_binary(body), do: body
  defp extract_message(_body), do: nil

  @spec transport_timeout?(term()) :: boolean()
  defp transport_timeout?(%Req.TransportError{reason: :timeout}), do: true
  defp transport_timeout?(:timeout), do: true
  defp transport_timeout?(_reason), do: false

  @spec transport_message(term()) :: String.t()
  defp transport_message(%Req.TransportError{} = error), do: Exception.message(error)
  defp transport_message(reason), do: "transport error: #{inspect(reason)}"

  @spec retryable?(type()) :: boolean()
  defp retryable?(type), do: type in [:rate_limit, :timeout, :server, :transport]
end
