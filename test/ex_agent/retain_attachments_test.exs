defmodule ExAgent.RetainAttachmentsTest do
  @moduledoc """
  `:retain_attachments` controls whether attachments from earlier turns are sent
  again on later requests.

  The default is `true`, which is the only correct answer for a conversation that
  refers back to a file ("now focus on the second document"). It is also what
  makes media caps surprising, because those caps are per *request*: the provider
  APIs are stateless, so every earlier attachment travels on every turn, and
  passing the same file twice puts the identical bytes in one request twice.
  """

  use ExUnit.Case, async: true

  alias ExAgent.Providers.OpenAI

  @png <<0x89, "PNG", 0x0D, 0x0A, 0x1A, 0x0A, 0::size(64)>>

  setup do
    a = Path.join(System.tmp_dir!(), "retain_a_#{System.unique_integer([:positive])}.png")
    b = Path.join(System.tmp_dir!(), "retain_b_#{System.unique_integer([:positive])}.png")
    File.write!(a, @png <> "A")
    File.write!(b, @png <> "B")
    on_exit(fn -> File.rm(a) && File.rm(b) end)
    %{a: a, b: b}
  end

  defp provider(plug) do
    %OpenAI{
      api_key: "sk-test",
      model: "gpt-4o",
      base_url: "http://x",
      modalities: [:text, :image],
      req: Req.new(plug: plug)
    }
  end

  defp recording_plug do
    test_pid = self()

    fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      send(test_pid, {:request, Jason.decode!(body)})

      Req.Test.json(conn, %{
        "choices" => [%{"message" => %{"role" => "assistant", "content" => "ok"}}]
      })
    end
  end

  # Every image part in a recorded request body, newest last.
  defp image_urls(body) do
    body["messages"]
    |> Enum.flat_map(fn message ->
      case message["content"] do
        parts when is_list(parts) -> parts
        _other -> []
      end
    end)
    |> Enum.filter(&(&1["type"] == "image_url"))
    |> Enum.map(&get_in(&1, ["image_url", "url"]))
  end

  defp drain_images(acc \\ []) do
    receive do
      {:request, body} -> drain_images([image_urls(body) | acc])
    after
      50 -> Enum.reverse(acc)
    end
  end

  defp next_images do
    receive do
      {:request, body} -> image_urls(body)
    after
      1_000 -> flunk("no request recorded")
    end
  end

  describe "retain_attachments: false" do
    test "given an earlier turn's file, then only the current turn's file is sent", ctx do
      {:ok, agent} =
        ExAgent.start_agent(provider: provider(recording_plug()), retain_attachments: false)

      ExAgent.chat(agent, "first", files: [%{path: ctx.a}])
      assert length(next_images()) == 1

      ExAgent.chat(agent, "second", files: [%{path: ctx.b}])
      assert [only] = next_images()
      assert String.ends_with?(only, Base.encode64(@png <> "B"))
    end

    test "given a turn with no files, then no attachments are sent at all", ctx do
      {:ok, agent} =
        ExAgent.start_agent(provider: provider(recording_plug()), retain_attachments: false)

      ExAgent.chat(agent, "first", files: [%{path: ctx.a}])
      assert length(next_images()) == 1

      ExAgent.chat(agent, "text only follow-up")
      assert next_images() == []
    end

    test "given repeated turns with files, then the count never grows", ctx do
      {:ok, agent} =
        ExAgent.start_agent(provider: provider(recording_plug()), retain_attachments: false)

      for _ <- 1..4, do: ExAgent.chat(agent, "look", files: [%{path: ctx.a}])

      assert Enum.map(1..4, fn _ -> length(next_images()) end) == [1, 1, 1, 1]
    end
  end

  describe "retain_attachments: true (the default)" do
    test "given no option, then earlier attachments still travel", ctx do
      {:ok, agent} = ExAgent.start_agent(provider: provider(recording_plug()))

      ExAgent.chat(agent, "first", files: [%{path: ctx.a}])
      assert length(next_images()) == 1

      ExAgent.chat(agent, "second", files: [%{path: ctx.b}])
      assert length(next_images()) == 2
    end

    test "given an explicit true, then behaviour matches the default", ctx do
      {:ok, agent} =
        ExAgent.start_agent(provider: provider(recording_plug()), retain_attachments: true)

      ExAgent.chat(agent, "first", files: [%{path: ctx.a}])
      _ = next_images()
      ExAgent.chat(agent, "second", files: [%{path: ctx.b}])
      assert length(next_images()) == 2
    end

    test "given the same file twice, then it is sent twice", ctx do
      {:ok, agent} = ExAgent.start_agent(provider: provider(recording_plug()))

      ExAgent.chat(agent, "first", files: [%{path: ctx.a}])
      _ = next_images()
      ExAgent.chat(agent, "again", files: [%{path: ctx.a}])

      urls = next_images()
      assert length(urls) == 2
      assert length(Enum.uniq(urls)) == 1
    end
  end

  describe "failures" do
    # Asserted through `Agent.start_link/1` because that is where the check lives:
    # via the DynamicSupervisor a raise arrives wrapped as {:error, {exception, _}}.
    test "given a non-boolean, then start_link raises rather than guessing" do
      base = [provider: provider(recording_plug())]

      assert_raise ArgumentError, ~r/:retain_attachments must be true or false/, fn ->
        ExAgent.Agent.start_link(base ++ [retain_attachments: "no"])
      end
    end

    test "given nil, then start_link raises rather than treating it as the default" do
      base = [provider: provider(recording_plug())]

      assert_raise ArgumentError, ~r/:retain_attachments must be true or false/, fn ->
        ExAgent.Agent.start_link(base ++ [retain_attachments: nil])
      end
    end

    test "given the option omitted, then it defaults to true", ctx do
      {:ok, agent} = ExAgent.start_agent(provider: provider(recording_plug()))

      ExAgent.chat(agent, "first", files: [%{path: ctx.a}])
      _ = next_images()
      ExAgent.chat(agent, "second", files: [%{path: ctx.b}])

      assert length(next_images()) == 2
    end
  end

  describe "edge cases" do
    test "given retain_attachments: false, then history still holds the attachments", ctx do
      {:ok, agent} =
        ExAgent.start_agent(provider: provider(recording_plug()), retain_attachments: false)

      ExAgent.chat(agent, "first", files: [%{path: ctx.a}])
      _ = next_images()
      ExAgent.chat(agent, "second", files: [%{path: ctx.b}])
      _ = next_images()

      attached =
        agent
        |> ExAgent.get_context()
        |> Map.fetch!(:messages)
        |> Enum.flat_map(& &1.attachments)

      # Stripping is a property of the request, not of the record of what was sent.
      assert length(attached) == 2
    end

    test "given a tool loop, then every turn strips earlier attachments", ctx do
      test_pid = self()
      counter = :counters.new(1, [])

      plug = fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        send(test_pid, {:request, Jason.decode!(body)})
        :counters.add(counter, 1, 1)

        message =
          case :counters.get(counter, 1) do
            1 ->
              %{
                "role" => "assistant",
                "content" => nil,
                "tool_calls" => [
                  %{"id" => "c1", "function" => %{"name" => "noop", "arguments" => "{}"}}
                ]
              }

            _final ->
              %{"role" => "assistant", "content" => "done"}
          end

        Req.Test.json(conn, %{"choices" => [%{"message" => message}]})
      end

      {:ok, tool} =
        ExAgent.Tool.new(
          name: "noop",
          description: "does nothing",
          parameters: %{"type" => "object", "properties" => %{}},
          function: fn _args -> {:ok, "ok"} end
        )

      {:ok, agent} =
        ExAgent.start_agent(
          provider: provider(plug),
          tools: [tool],
          retain_attachments: false
        )

      ExAgent.chat(agent, "first", files: [%{path: ctx.a}])
      ExAgent.chat(agent, "second", files: [%{path: ctx.b}])

      # The first turn spends two provider calls (tool request, then the answer),
      # so drain every request rather than assuming one per turn.
      requests = drain_images()

      # Three or four calls, and not one of them carries more than a single image.
      assert length(requests) >= 3
      assert Enum.all?(requests, &(length(&1) <= 1))

      # The last call belongs to turn 2, so it must carry B and nothing else.
      assert [last] = List.last(requests)
      assert String.ends_with?(last, Base.encode64(@png <> "B"))
    end

    test "given a stream, then earlier attachments are stripped too", ctx do
      test_pid = self()

      plug = fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        send(test_pid, {:request, Jason.decode!(body)})

        conn
        |> Plug.Conn.put_resp_content_type("text/event-stream")
        |> Plug.Conn.send_resp(200, "data: [DONE]\n\n")
      end

      {:ok, agent} = ExAgent.start_agent(provider: provider(plug), retain_attachments: false)

      agent |> ExAgent.chat_stream("first", files: [%{path: ctx.a}]) |> Enum.to_list()
      assert length(next_images()) == 1

      agent |> ExAgent.chat_stream("second", files: [%{path: ctx.b}]) |> Enum.to_list()
      assert [only] = next_images()
      assert String.ends_with?(only, Base.encode64(@png <> "B"))
    end

    test "given an uploaded file_ref, then it is stripped like inline bytes", ctx do
      ref = %ExAgent.FileRef{
        provider: :openai,
        file_id: "file-123",
        mime_type: "application/pdf"
      }

      provider = %{provider(recording_plug()) | modalities: [:text, :image, :document]}
      {:ok, agent} = ExAgent.start_agent(provider: provider, retain_attachments: false)

      ExAgent.chat(agent, "read this", files: [%{file_ref: ref}])
      _ = next_images()

      ExAgent.chat(agent, "now this image", files: [%{path: ctx.a}])

      receive do
        {:request, body} ->
          parts =
            body["messages"]
            |> Enum.flat_map(fn m -> if is_list(m["content"]), do: m["content"], else: [] end)

          assert Enum.count(parts, &(&1["type"] == "image_url")) == 1
          refute Enum.any?(parts, &(&1["type"] == "file"))
      end
    end
  end
end
