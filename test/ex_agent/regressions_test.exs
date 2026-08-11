defmodule ExAgent.RegressionsTest do
  @moduledoc """
  One test per bug found in the end-to-end audit of v0.3.0.

  Each of these shipped green: the suite covered the *shape* of every code path
  but not the behaviour a user would actually observe. They are grouped here,
  next to the symptom, so a future refactor that reintroduces one fails loudly.
  """

  use ExUnit.Case, async: true

  alias ExAgent.{Chunk, Error, Message, Response, Skill, Tool}
  alias ExAgent.Patterns.{Router, Skills, Subagents}
  alias ExAgent.Providers.{Gemini, OpenAI}
  alias ExAgent.Services.{GeminiService, OpenAIDialect, OpenAIService}
  alias ExAgent.Test.FakeSSE

  # --- Helpers ---

  defp user_msg(content \\ "hi") do
    {:ok, message} = Message.new(role: :user, content: content)
    message
  end

  defp openai(plug),
    do: %OpenAI{api_key: "sk-test", base_url: "http://x", req: Req.new(plug: plug)}

  defp gemini(plug),
    do: %Gemini{
      api_key: "k",
      model: "gemini-3.6-flash",
      base_url: "http://x",
      req: Req.new(plug: plug)
    }

  # Answers every request with `body`, forwarding the decoded request to the test.
  defp echo(body) do
    test = self()

    fn conn ->
      {:ok, raw, conn} = Plug.Conn.read_body(conn)
      send(test, {:request, Jason.decode!(raw)})
      Req.Test.json(conn, body)
    end
  end

  defp sent_body do
    receive do
      {:request, body} -> body
    after
      500 -> flunk("no request was made")
    end
  end

  defp chat_reply(content),
    do: %{"choices" => [%{"message" => %{"role" => "assistant", "content" => content}}]}

  defp start_agent!(opts) do
    {:ok, agent} = ExAgent.start_agent(opts)
    on_exit(fn -> if Process.alive?(agent), do: ExAgent.stop_agent(agent) end)
    agent
  end

  describe "the stream transport and the caller's mailbox" do
    # `chat_stream/3` runs in the *calling* process. A bare `receive` matched
    # anything, and `Req.parse_message/2` answering `:unknown` dropped it - so
    # streaming inside a LiveView or GenServer silently ate that process's own
    # messages and the matching handle_info never fired.
    test "given unrelated messages in the mailbox, then streaming leaves them untouched" do
      send(self(), {:phoenix, :diff})
      send(self(), :tick)

      sse = ~s(data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n)

      provider = %OpenAI{
        api_key: "k",
        base_url: "http://x",
        req: Req.new(adapter: FakeSSE.adapter([sse]))
      }

      assert [%Chunk{type: :text_delta, text: "hi"}, %Chunk{type: :done}] =
               provider |> OpenAIService.stream([user_msg()]) |> Enum.to_list()

      assert_received {:phoenix, :diff}
      assert_received :tick
    end

    test "given a foreign message arriving mid-stream, then the stream still completes" do
      parts = [
        ~s(data: {"choices":[{"delta":{"content":"a"}}]}\n\n),
        ~s(data: {"choices":[{"delta":{"content":"b"}}]}\n\n),
        "data: [DONE]\n\n"
      ]

      provider = %OpenAI{
        api_key: "k",
        base_url: "http://x",
        req: Req.new(adapter: FakeSSE.adapter(parts))
      }

      send(self(), :interleaved)

      texts =
        provider
        |> OpenAIService.stream([user_msg()])
        |> Enum.filter(&(&1.type == :text_delta))
        |> Enum.map(& &1.text)

      assert texts == ["a", "b"]
      assert_received :interleaved
    end
  end

  describe "Gemini response parsing" do
    # A reasoning part matched the "first text part" clause, so the model's
    # scratchpad was returned as the answer and the answer was discarded.
    test "given a thought part before the answer, then the answer is the content" do
      body = %{
        "candidates" => [
          %{
            "content" => %{
              "parts" => [
                %{"text" => "let me think...", "thought" => true},
                %{"text" => "THE ANSWER"}
              ]
            },
            "finishReason" => "STOP"
          }
        ]
      }

      assert {:ok, response} = GeminiService.chat(gemini(echo(body)), [user_msg()])
      assert response.content == "THE ANSWER"
      assert response.thinking == "let me think..."
      refute response.message.content =~ "think"
    end

    test "given text split across parts, then every piece is kept" do
      body = %{
        "candidates" => [
          %{"content" => %{"parts" => [%{"text" => "Hello "}, %{"text" => "world"}]}}
        ]
      }

      assert {:ok, %Response{content: "Hello world"}} =
               GeminiService.chat(gemini(echo(body)), [user_msg()])
    end

    test "given several function calls, then all of them come back" do
      body = %{
        "candidates" => [
          %{
            "content" => %{
              "parts" => [
                %{"functionCall" => %{"name" => "a", "args" => %{"x" => 1}}},
                %{"functionCall" => %{"name" => "b", "args" => %{}}}
              ]
            }
          }
        ]
      }

      assert {:tool_calls, [%{"name" => "a", "args" => %{"x" => 1}}, %{"name" => "b"}]} =
               GeminiService.chat(gemini(echo(body)), [user_msg()])
    end

    test "given a candidate with no usable parts, then it is an error, not a crash" do
      body = %{"candidates" => [%{"content" => %{"parts" => [%{"inlineData" => %{}}]}}]}

      assert {:error, %Error{type: :server}} =
               GeminiService.chat(gemini(echo(body)), [user_msg()])
    end
  end

  describe "Gemini generation config" do
    # `prepare_opts/2` merged `:max_tokens` while the body builder read
    # `:max_output_tokens`, so the ceiling was never sent and every Gemini
    # response silently used the API default.
    test "given a provider max_tokens, then it reaches generationConfig" do
      provider = %{gemini(echo(chat_reply(nil))) | max_tokens: 4096}
      GeminiService.chat(provider, [user_msg()])

      assert %{"generationConfig" => %{"maxOutputTokens" => 4096}} = sent_body()
    end

    test "given a per-call max_tokens, then it wins over the provider's" do
      provider = %{gemini(echo(chat_reply(nil))) | max_tokens: 100}
      GeminiService.chat(provider, [user_msg()], max_tokens: 8000)

      assert %{"generationConfig" => %{"maxOutputTokens" => 8000}} = sent_body()
    end

    test "given Google's own spelling, then it is honoured too" do
      GeminiService.chat(gemini(echo(chat_reply(nil))), [user_msg()], max_output_tokens: 77)

      assert %{"generationConfig" => %{"maxOutputTokens" => 77}} = sent_body()
    end

    test "given no ceiling anywhere, then none is sent and the model's default applies" do
      GeminiService.chat(gemini(echo(chat_reply(nil))), [user_msg()])

      refute match?(%{"generationConfig" => %{"maxOutputTokens" => _}}, sent_body())
    end
  end

  describe "parallel tool calls" do
    test "given two calls in one turn, then both are returned with their own ids" do
      body = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{"id" => "call_1", "function" => %{"name" => "a", "arguments" => "{}"}},
                %{"id" => "call_2", "function" => %{"name" => "b", "arguments" => "{}"}}
              ]
            }
          }
        ]
      }

      assert {:tool_calls,
              [%{"id" => "call_1", "name" => "a"}, %{"id" => "call_2", "name" => "b"}]} =
               OpenAIDialect.parse_response(body, OpenAI)
    end

    # The assistant message used to be rebuilt with `id = name`, discarding the
    # API's id - so calling one tool twice produced two colliding ids.
    test "given a recorded call, then the provider's id is what goes back on the wire" do
      {:ok, assistant} =
        Message.new(
          role: :assistant,
          content: "",
          tool_calls: [%{"id" => "call_abc", "name" => "search", "args" => %{}}]
        )

      body = OpenAIDialect.build_body("gpt-4o", [assistant], [])
      [%{"tool_calls" => [call]}] = body["messages"]

      assert call["id"] == "call_abc"
      assert call["function"]["name"] == "search"
    end

    test "given the agent runs two calls, then each gets its own tool result" do
      calls = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{
                  "id" => "call_1",
                  "function" => %{"name" => "double", "arguments" => ~s({"n":2})}
                },
                %{
                  "id" => "call_2",
                  "function" => %{"name" => "double", "arguments" => ~s({"n":5})}
                }
              ]
            }
          }
        ]
      }

      {:ok, counter} = Elixir.Agent.start_link(fn -> 0 end)

      plug = fn conn ->
        turn = Elixir.Agent.get_and_update(counter, &{&1, &1 + 1})
        Req.Test.json(conn, if(turn == 0, do: calls, else: chat_reply("done")))
      end

      tool = %Tool{
        name: "double",
        description: "doubles",
        parameters: %{},
        function: fn %{"n" => n} -> %{"result" => n * 2} end
      }

      agent = start_agent!(provider: openai(plug), tools: [tool])

      assert {:ok, %Response{content: "done"}} = ExAgent.chat(agent, "go")

      results =
        agent
        |> ExAgent.get_context()
        |> Map.fetch!(:messages)
        |> Enum.filter(&(&1.role == :tool))

      assert length(results) == 2
      assert Enum.map(results, & &1.tool_call_id) == ["call_1", "call_2"]
      assert Enum.map(results, & &1.content) == [~s({"result":4}), ~s({"result":10})]
    end
  end

  describe "tool results that are not strings" do
    # `to_string/1` on a map raised Protocol.UndefinedError, killing the task and
    # surfacing as an opaque :server error - for the most natural tool shape there is.
    test "given a tool returning a map, then it is encoded rather than crashing the turn" do
      call = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{"id" => "c1", "function" => %{"name" => "t", "arguments" => "{}"}}
              ]
            }
          }
        ]
      }

      {:ok, counter} = Elixir.Agent.start_link(fn -> 0 end)

      plug = fn conn ->
        turn = Elixir.Agent.get_and_update(counter, &{&1, &1 + 1})
        Req.Test.json(conn, if(turn == 0, do: call, else: chat_reply("ok")))
      end

      tool = %Tool{
        name: "t",
        description: "d",
        parameters: %{},
        function: fn _args -> %{"total" => 3} end
      }

      agent = start_agent!(provider: openai(plug), tools: [tool])

      assert {:ok, %Response{content: "ok"}} = ExAgent.chat(agent, "go")

      assert [%Message{content: ~s({"total":3})}] =
               agent
               |> ExAgent.get_context()
               |> Map.fetch!(:messages)
               |> Enum.filter(&(&1.role == :tool))
    end
  end

  describe "skill deactivation" do
    test "given a skill that stops matching, then the agent's own prompt returns" do
      {:ok, skill} =
        Skill.new(
          name: "sql",
          system_prompt: "YOU ARE A SQL EXPERT",
          activation_fn: fn context -> List.last(context.messages).content =~ "sql" end
        )

      provider = %{openai(echo(chat_reply("ok"))) | system_prompt: "BASE PROMPT"}
      agent = start_agent!(provider: provider, skills: [skill])

      ExAgent.chat(agent, "write me sql")

      assert %{"role" => "system", "content" => "YOU ARE A SQL EXPERT"} =
               hd(sent_body()["messages"])

      ExAgent.chat(agent, "now tell me a joke")
      assert %{"role" => "system", "content" => "BASE PROMPT"} = hd(sent_body()["messages"])
    end

    test "given a provider with no system_prompt field, then applying a skill does not raise" do
      state = %{provider: %{__struct__: NoPrompt, tools: []}, active_skill: nil}
      {:ok, skill} = Skill.new(name: "s", system_prompt: "p")

      assert %{active_skill: ^skill} = Skills.apply_skill(state, skill)
    end

    test "given no skill was ever active, then clearing is a no-op" do
      state = %{provider: %{__struct__: NoPrompt}, active_skill: nil}

      assert Skills.clear_skill(state) == state
    end
  end

  describe "patterns survive a failing agent" do
    defmodule Crasher do
      @moduledoc false
      use GenServer

      def start_link(_opts), do: GenServer.start_link(__MODULE__, nil)

      @impl true
      def init(_arg), do: {:ok, nil}

      @impl true
      def handle_call(_request, _from, _state), do: raise("boom")
    end

    # One crashed agent used to raise a CaseClauseError in the caller, taking
    # down the routes that had already answered.
    test "given a crashing agent, then the route reports it instead of taking down the caller" do
      {:ok, crasher} = Crasher.start_link([])
      Process.flag(:trap_exit, true)

      routes = [%{name: "broken", agent: crasher, match_fn: fn _input -> true end}]

      assert {:ok, output} = Router.run("x", routes: routes)
      assert output =~ "broken"
      assert output =~ "Error"
    end

    test "given a subagent whose provider fails, then the failure is named, not raised" do
      spec = %{
        name: "helper",
        description: "d",
        provider: openai(fn conn -> Plug.Conn.send_resp(conn, 500, "{}") end),
        system_prompt: nil,
        tools: []
      }

      assert [{"helper", {:error, %Error{}}}] =
               Subagents.run([{spec, "hi"}])
    end

    test "given a subagent tool called without a query, then it answers instead of crashing" do
      spec = %{
        name: "helper",
        description: "d",
        provider: openai(echo(chat_reply("x"))),
        tools: []
      }

      [tool] = Subagents.tools([spec])

      assert {:error, message} = tool.function.(%{})
      assert message =~ "requires a \"query\""
    end
  end

  describe "unparseable responses" do
    test "given a message with neither content nor tool calls, then it is a normalized error" do
      body = %{"choices" => [%{"message" => %{"refusal" => "I cannot help"}}]}

      assert {:error, %Error{type: :server}} = OpenAIDialect.parse_response(body, OpenAI)
    end
  end

  describe "credentials are not printed" do
    test "given a provider struct, then inspecting it redacts the key and the client" do
      dumped = inspect(OpenAI.new(api_key: "sk-super-secret-123"), limit: :infinity)

      refute dumped =~ "sk-super-secret-123"
      assert dumped =~ "..."
    end

    test "given a compatible provider, then gateway headers are redacted too" do
      provider =
        ExAgent.Providers.OpenAICompatible.new(
          base_url: "http://x",
          model: "m",
          headers: [{"Modal-Key", "wk-secret"}]
        )

      refute inspect(provider, limit: :infinity) =~ "wk-secret"
    end
  end

  describe "no invented generation defaults" do
    # `max_tokens: 512` truncated most real answers at `finish_reason: :length`,
    # and `temperature: 0.6` broke models that reject the parameter outright.
    test "given nothing configured, then neither temperature nor max_tokens is sent" do
      OpenAIService.chat(openai(echo(chat_reply("ok"))), [user_msg()])
      body = sent_body()

      refute Map.has_key?(body, "temperature")
      refute Map.has_key?(body, "max_tokens")
    end

    test "given an integer temperature, then it is accepted rather than rejected" do
      assert %OpenAI{temperature: 1} = OpenAI.new(api_key: "k", temperature: 1)
      assert %Gemini{temperature: 1} = Gemini.new(api_key: "k", temperature: 1)
    end
  end

  describe "tool iteration ceiling" do
    test "given :max_tool_iterations, then the agent stops there" do
      {:ok, counter} = Elixir.Agent.start_link(fn -> 0 end)

      call = %{
        "choices" => [
          %{
            "message" => %{
              "role" => "assistant",
              "content" => nil,
              "tool_calls" => [
                %{"id" => "c", "function" => %{"name" => "loop", "arguments" => "{}"}}
              ]
            }
          }
        ]
      }

      plug = fn conn ->
        Elixir.Agent.update(counter, &(&1 + 1))
        Req.Test.json(conn, call)
      end

      tool = %Tool{
        name: "loop",
        description: "d",
        parameters: %{},
        function: fn _args -> :again end
      }

      agent = start_agent!(provider: openai(plug), tools: [tool], max_tool_iterations: 2)

      assert {:error, :max_tool_iterations_reached} = ExAgent.chat(agent, "go")
      assert Elixir.Agent.get(counter, & &1) == 2
    end
  end

  describe "observability" do
    test "given a chat call, then a telemetry span carries model, result, and tokens" do
      :telemetry.attach(
        "regressions-chat",
        [:ex_agent, :chat, :stop],
        fn _event, measurements, metadata, pid -> send(pid, {:span, measurements, metadata}) end,
        self()
      )

      on_exit(fn -> :telemetry.detach("regressions-chat") end)

      body = %{
        "choices" => [%{"message" => %{"role" => "assistant", "content" => "hi"}}],
        "usage" => %{"prompt_tokens" => 3, "completion_tokens" => 2, "total_tokens" => 5}
      }

      provider = %{openai(echo(body)) | model: "gpt-4o"}
      assert {:ok, _response} = ExAgent.Provider.chat(provider, [user_msg()])

      assert_received {:span, measurements, metadata}
      assert measurements.total_tokens == 5
      assert measurements.duration > 0
      assert metadata.model == "gpt-4o"
      assert metadata.result == :ok
      assert metadata.provider == OpenAI
    end

    test "given a failing call, then the span names the error type and retryability" do
      :telemetry.attach(
        "regressions-error",
        [:ex_agent, :chat, :stop],
        fn _event, _measurements, metadata, pid -> send(pid, {:span, metadata}) end,
        self()
      )

      on_exit(fn -> :telemetry.detach("regressions-error") end)

      provider = openai(fn conn -> Plug.Conn.send_resp(conn, 429, "{}") end)
      assert {:error, _error} = ExAgent.Provider.chat(provider, [user_msg()])

      assert_received {:span, %{result: :error, error_type: :rate_limit, retryable?: true}}
    end
  end

  describe "bounded history" do
    test "given :max_history, then the oldest turns are dropped as the chat continues" do
      agent = start_agent!(provider: openai(echo(chat_reply("ok"))), max_history: 2)

      ExAgent.chat(agent, "one")
      ExAgent.chat(agent, "two")
      ExAgent.chat(agent, "three")

      messages = agent |> ExAgent.get_context() |> Map.fetch!(:messages)

      assert length(messages) == 2
      assert Enum.map(messages, & &1.role) == [:user, :assistant]
      assert hd(messages).content == "three"
    end

    test "given no :max_history, then nothing is dropped" do
      agent = start_agent!(provider: openai(echo(chat_reply("ok"))))

      ExAgent.chat(agent, "one")
      ExAgent.chat(agent, "two")

      assert length(ExAgent.get_context(agent).messages) == 4
    end

    test "given a window that would orphan a tool result, then the result is dropped too" do
      {:ok, assistant} =
        Message.new(role: :assistant, content: "", tool_calls: [%{"name" => "t", "args" => %{}}])

      {:ok, result} = Message.new(role: :tool, content: "42", tool_call_id: "t")
      {:ok, answer} = Message.new(role: :assistant, content: "done")

      context = ExAgent.Context.new(messages: [assistant, result, answer])

      # Keeping two would start the window at the orphaned :tool message, which
      # every provider rejects.
      assert [%Message{role: :assistant, content: "done"}] =
               ExAgent.Context.trim(context, 2).messages
    end

    test "given system messages, then they survive the window" do
      {:ok, system} = Message.new(role: :system, content: "be terse")
      {:ok, old} = Message.new(role: :user, content: "old")
      {:ok, new} = Message.new(role: :user, content: "new")

      context = ExAgent.Context.new(messages: [system, old, new])

      assert [%Message{role: :system}, %Message{content: "new"}] =
               ExAgent.Context.trim(context, 1).messages
    end
  end

  describe "embeddings batch limits" do
    test "given more inputs than OpenAI accepts, then the limit is named up front" do
      provider = openai(fn conn -> Req.Test.json(conn, %{"data" => []}) end)
      inputs = Enum.map(1..2049, &"chunk #{&1}")

      assert {:error, %Error{type: :invalid_request} = error} = ExAgent.embed(provider, inputs)
      assert error.message =~ "at most 2048"
      assert error.message =~ "chunk_every"
    end
  end
end
