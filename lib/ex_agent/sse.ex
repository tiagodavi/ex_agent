defmodule ExAgent.SSE do
  @moduledoc """
  Server-Sent Events framing.

  Decoding is separated from transport because a frame can arrive split across
  two TCP reads: `decode/1` returns only the frames that are complete, plus the
  remainder to prepend to the next read.

      {frames, rest} = ExAgent.SSE.decode(buffer <> incoming)
      # keep `rest` as the new buffer

  A `data: [DONE]` sentinel decodes to the `:done` atom, so termination is a
  decoding concern rather than something each provider re-implements.
  """

  @type frame :: map() | :done

  # The SSE spec terminates lines with CRLF, LF, or a bare CR, and providers
  # disagree: OpenAI sends LF, Gemini sends CRLF. Matching only "\n\n" finds no
  # boundary at all in a CRLF stream, so every frame stays buffered and the
  # stream yields nothing. CRLF is listed first so it wins over a bare CR.
  @event_separator ~r/\r\n\r\n|\n\n|\r\r/
  @line_separator ~r/\r\n|\n|\r/

  @doc """
  Decodes a buffer into complete frames plus the unconsumed remainder.

  Comment and keep-alive events (those with no `data:` line) and frames whose
  payload is not valid JSON are skipped rather than raising - a malformed frame
  should not tear down a stream that is otherwise fine.

  ## Examples

      iex> ExAgent.SSE.decode(~s(data: {"a":1}\\n\\ndata: [DONE]\\n\\n))
      {[%{"a" => 1}, :done], ""}

      iex> ExAgent.SSE.decode(~s(data: {"a":1}\\n\\ndata: {"b":))
      {[%{"a" => 1}], ~s(data: {"b":)}
  """
  @spec decode(binary()) :: {[frame()], binary()}
  def decode(buffer) do
    {events, rest} = take_events(buffer)
    {Enum.flat_map(events, &decode_event/1), rest}
  end

  @doc """
  Splits an SSE buffer into complete events plus the trailing partial event.

  Events are separated by a blank line; the second element is always the
  (possibly empty) incomplete remainder, held back for the next read.

  ## Examples

      iex> ExAgent.SSE.take_events("a\\n\\nb\\n\\nc")
      {["a", "b"], "c"}
  """
  @spec take_events(binary()) :: {[binary()], binary()}
  def take_events(buffer) do
    parts = String.split(buffer, @event_separator)
    {complete, [rest]} = Enum.split(parts, -1)
    {complete, rest}
  end

  @doc """
  Extracts and joins the payload of every `data:` line in one SSE event.

  Returns `:done` for a `[DONE]` sentinel and `nil` when the event carries no
  data.

  ## Examples

      iex> ExAgent.SSE.event_data("event: message\\ndata: hello")
      "hello"

      iex> ExAgent.SSE.event_data("data: [DONE]")
      :done

      iex> ExAgent.SSE.event_data(": keep-alive")
      nil
  """
  @spec event_data(binary()) :: binary() | :done | nil
  def event_data(event) do
    data =
      event
      |> String.split(@line_separator)
      |> Enum.filter(&String.starts_with?(&1, "data:"))
      # The spec joins consecutive `data:` lines with a newline and strips a
      # single leading space, not all whitespace - a JSON payload survives
      # either way, but a plain-text stream does not.
      |> Enum.map_join("\n", fn line ->
        line |> String.replace_prefix("data:", "") |> String.replace_prefix(" ", "")
      end)

    case data do
      "" -> nil
      "[DONE]" -> :done
      json -> json
    end
  end

  @spec decode_event(binary()) :: [frame()]
  defp decode_event(event) do
    case event_data(event) do
      nil ->
        []

      :done ->
        [:done]

      json ->
        case Jason.decode(json) do
          {:ok, decoded} -> [decoded]
          {:error, _reason} -> []
        end
    end
  end
end
