defmodule ExAgent.Patterns.WorkflowsTest do
  @moduledoc """
  The four workflow patterns: Chain, Reflection, MapReduce, Consensus.

  Each is exercised through a stubbed provider so the *control flow* is what is
  being tested — how many calls happen, in what order, and what the pattern does
  when one of them fails.
  """

  use ExUnit.Case, async: true

  alias ExAgent.Patterns.{Chain, Consensus, MapReduce, Reflection}
  alias ExAgent.Providers.OpenAI

  # --- Stubs ---

  # Answers with `reply_fun.(prompt)`, recording every prompt it saw.
  defp provider(reply_fun) do
    test_pid = self()

    plug = fn conn ->
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      prompt = body |> Jason.decode!() |> get_in(["messages", Access.at(-1), "content"])
      send(test_pid, {:prompt, prompt})

      case reply_fun.(prompt) do
        {:error, status} ->
          Plug.Conn.send_resp(conn, status, ~s({"error":{"message":"nope"}}))

        reply ->
          Req.Test.json(conn, %{
            "choices" => [%{"message" => %{"role" => "assistant", "content" => reply}}]
          })
      end
    end

    %OpenAI{api_key: "sk-test", model: "gpt-4o", base_url: "http://x", req: Req.new(plug: plug)}
  end

  defp echoing(reply), do: provider(fn _prompt -> reply end)

  defp prompts do
    Enum.reverse(drain([]))
  end

  defp drain(acc) do
    receive do
      {:prompt, prompt} -> drain([prompt | acc])
    after
      0 -> acc
    end
  end

  # Replies differently on each call, so a loop can be observed progressing.
  defp scripted(replies) do
    {:ok, counter} = Agent.start_link(fn -> 0 end)

    provider(fn _prompt ->
      index = Agent.get_and_update(counter, &{&1, &1 + 1})
      Enum.at(replies, index, List.last(replies))
    end)
  end

  describe "Chain.run/2" do
    test "given several steps, then each sees the previous output" do
      steps = [
        Chain.llm(echoing("extracted"), fn input -> "extract from #{input}" end),
        Chain.llm(echoing("checklist"), fn input -> "format #{input}" end)
      ]

      assert {:ok, "checklist"} = Chain.run("transcript", steps: steps)
      assert ["extract from transcript", "format extracted"] = prompts()
    end

    test "given a plain function step, then it transforms without an LLM call" do
      steps = [
        fn input -> {:ok, String.upcase(input)} end,
        Chain.llm(echoing("done"), fn input -> "use #{input}" end)
      ]

      assert {:ok, "done"} = Chain.run("hi", steps: steps)
      assert ["use HI"] = prompts()
    end

    test "given no steps, then the input passes through untouched" do
      assert {:ok, "unchanged"} = Chain.run("unchanged", steps: [])
    end

    # A gate declining is not a failure — it is the point of having gates.
    test "given a halting gate, then later steps never run" do
      steps = [
        Chain.llm(echoing("NO"), fn input -> "is #{input} a bug?" end),
        fn answer -> if answer == "YES", do: {:ok, answer}, else: {:halt, :not_a_bug} end,
        Chain.llm(echoing("never"), fn _input -> "should not be sent" end)
      ]

      assert {:halted, :not_a_bug} = Chain.run("ticket", steps: steps)
      assert length(prompts()) == 1
    end

    test "given a failing step, then the error names which step failed" do
      steps = [
        Chain.llm(echoing("ok"), fn _input -> "first" end),
        Chain.llm(provider(fn _prompt -> {:error, 500} end), fn _input -> "second" end),
        Chain.llm(echoing("never"), fn _input -> "third" end)
      ]

      assert {:error, {1, %ExAgent.Error{type: :server}}} = Chain.run("in", steps: steps)
      assert length(prompts()) == 2
    end

    test "given a step returning garbage, then it is reported rather than treated as success" do
      assert {:error, {0, {:invalid_step_result, :whoops}}} =
               Chain.run("in", steps: [fn _input -> :whoops end])
    end

    test "given a tool-calling provider, then the step fails instead of passing a tuple on" do
      plug = fn conn ->
        Req.Test.json(conn, %{
          "choices" => [
            %{
              "message" => %{
                "role" => "assistant",
                "content" => nil,
                "tool_calls" => [
                  %{"id" => "c", "function" => %{"name" => "t", "arguments" => "{}"}}
                ]
              }
            }
          ]
        })
      end

      tool_caller = %OpenAI{api_key: "k", base_url: "http://x", req: Req.new(plug: plug)}

      assert {:error, {0, {:unexpected_tool_calls, ["t"]}}} =
               Chain.run("in", steps: [Chain.llm(tool_caller, fn _input -> "go" end)])
    end
  end

  describe "Reflection.run/2" do
    test "given a critic that approves at once, then no revision happens" do
      generator = echoing("draft")
      critic = echoing("APPROVED")

      assert {:ok, result} = Reflection.run("write it", generator: generator, critic: critic)

      assert result.output == "draft"
      assert result.rounds == 0
      assert result.accepted?
      assert length(result.critiques) == 1
    end

    test "given a critic that approves on the second look, then one revision happens" do
      # draft, critique, revision, approval
      provider = scripted(["first draft", "too short", "second draft", "APPROVED"])

      assert {:ok, result} = Reflection.run("write it", generator: provider)

      assert result.output == "second draft"
      assert result.rounds == 1
      assert result.critiques == ["too short", "APPROVED"]
    end

    test "given a custom accept?, then it decides when to stop" do
      provider = scripted(["draft", "score: 9/10"])

      assert {:ok, result} =
               Reflection.run("write it",
                 generator: provider,
                 accept?: &String.contains?(&1, "9/10")
               )

      assert result.accepted?
    end

    # An LLM critic can always find something to complain about, so the ceiling is
    # the difference between a workflow and a runaway bill.
    test "given a critic that never approves, then it stops at max_rounds unaccepted" do
      provider = echoing("still wrong")

      assert {:max_rounds, result} =
               Reflection.run("write it", generator: provider, max_rounds: 2)

      refute result.accepted?
      assert result.rounds == 2
      assert length(result.critiques) == 2
      assert result.output == "still wrong"
    end

    test "given max_rounds, then the outcome is not {:ok, _} — unapproved work is not passed off as approved" do
      assert {:max_rounds, _result} =
               Reflection.run("t", generator: echoing("nope"), max_rounds: 1)
    end

    test "given a failing generator, then the error surfaces" do
      assert {:error, %ExAgent.Error{type: :server}} =
               Reflection.run("t", generator: provider(fn _prompt -> {:error, 500} end))
    end

    test "given a failing critic, then the error surfaces" do
      generator = echoing("draft")
      critic = provider(fn _prompt -> {:error, 503} end)

      assert {:error, %ExAgent.Error{}} =
               Reflection.run("t", generator: generator, critic: critic)
    end

    test "given no critic, then the generator reviews its own work" do
      provider = scripted(["draft", "APPROVED"])

      assert {:ok, %{accepted?: true}} = Reflection.run("t", generator: provider)
    end
  end

  describe "MapReduce.run/3" do
    test "given sections, then each is mapped and the outputs reduced" do
      provider = echoing("summary")

      assert {:ok, result} =
               MapReduce.run(["a", "b", "c"], provider,
                 map: &"summarise #{&1}",
                 reduce: fn outputs -> {:ok, Enum.join(outputs, "|")} end
               )

      assert result.output == "summary|summary|summary"
      assert result.sections == 3
      assert result.failures == []
      assert length(prompts()) == 3
    end

    test "given a model reducer, then it receives the mapped outputs" do
      assert {:ok, result} =
               MapReduce.run(["a", "b"], echoing("part"),
                 map: &"map #{&1}",
                 reduce: {echoing("merged"), fn parts -> "merge #{Enum.join(parts, ",")}" end}
               )

      assert result.output == "merged"
      assert "merge part,part" in prompts()
    end

    test "given no sections, then the reducer still runs on an empty list" do
      assert {:ok, %{output: "none", sections: 0}} =
               MapReduce.run([], echoing("x"),
                 map: & &1,
                 reduce: fn [] -> {:ok, "none"} end
               )
    end

    # A summary built from 2 of 3 sections is usually worth having, but never worth
    # mistaking for all 3.
    test "given one failing section, then the rest still reduce and the failure is reported" do
      provider =
        provider(fn prompt -> if String.contains?(prompt, "b"), do: {:error, 500}, else: "ok" end)

      assert {:ok, result} =
               MapReduce.run(["a", "b", "c"], provider,
                 map: &"section #{&1}",
                 reduce: fn outputs -> {:ok, length(outputs)} end
               )

      assert result.output == 2
      assert result.sections == 2
      assert [{1, %ExAgent.Error{}}] = result.failures
    end

    test "given every section failing, then it does not silently reduce nothing" do
      provider = provider(fn _prompt -> {:error, 500} end)

      assert {:error, :all_sections_failed} =
               MapReduce.run(["a", "b"], provider,
                 map: & &1,
                 reduce: fn _outputs -> {:ok, :never} end
               )
    end

    test "given a failing reducer, then the error surfaces" do
      assert {:error, :bad_merge} =
               MapReduce.run(["a"], echoing("ok"),
                 map: & &1,
                 reduce: fn _outputs -> {:error, :bad_merge} end
               )
    end

    test "given a reducer returning garbage, then it is reported" do
      assert {:error, {:invalid_reduce_result, :oops}} =
               MapReduce.run(["a"], echoing("ok"), map: & &1, reduce: fn _outputs -> :oops end)
    end
  end

  describe "Consensus.run/2" do
    test "given a unanimous vote, then agreement is 1.0" do
      assert {:ok, verdict} = Consensus.run("q", voters: echoing("YES"), samples: 3)

      assert verdict.answer == "yes"
      assert verdict.agreement == 1.0
      assert verdict.votes == %{"yes" => 3}
      assert length(verdict.answers) == 3
    end

    test "given a split vote, then the majority wins and agreement reports the split" do
      provider = scripted(["YES", "NO", "YES"])

      assert {:ok, verdict} = Consensus.run("q", voters: provider, samples: 3)

      assert verdict.answer == "yes"
      assert_in_delta verdict.agreement, 2 / 3, 1.0e-9
      assert verdict.votes == %{"yes" => 2, "no" => 1}
    end

    test "given a list of voters, then each is asked once" do
      voters = [echoing("A"), echoing("A"), echoing("B")]

      assert {:ok, verdict} = Consensus.run("q", voters: voters)

      assert verdict.answer == "a"
      assert length(verdict.answers) == 3
    end

    test "given differing formatting, then the default normalization still agrees" do
      provider = scripted(["YES", " yes ", "Yes"])

      assert {:ok, %{agreement: 1.0, answer: "yes"}} =
               Consensus.run("q", voters: provider, samples: 3)
    end

    test "given a custom normalize, then votes are counted on its output" do
      provider = scripted(["Definitely yes", "I think yes", "No way"])

      assert {:ok, verdict} =
               Consensus.run("q",
                 voters: provider,
                 samples: 3,
                 normalize: &String.contains?(String.downcase(&1), "yes")
               )

      assert verdict.answer == true
      assert_in_delta verdict.agreement, 2 / 3, 1.0e-9
    end

    test "given every voter failing, then there is no answer to report" do
      assert {:error, :no_answers} =
               Consensus.run("q", voters: provider(fn _prompt -> {:error, 500} end), samples: 2)
    end

    test "given some voters failing, then the survivors decide and failures are kept" do
      {:ok, counter} = Agent.start_link(fn -> 0 end)

      provider =
        provider(fn _prompt ->
          if Agent.get_and_update(counter, &{&1, &1 + 1}) == 0, do: {:error, 500}, else: "YES"
        end)

      assert {:ok, verdict} = Consensus.run("q", voters: provider, samples: 3)

      assert verdict.answer == "yes"
      assert length(verdict.failures) == 1
      # Agreement is over the votes actually cast, not over the voters asked.
      assert verdict.agreement == 1.0
    end

    test "given a tie, then the earliest answer wins deterministically" do
      # A map has no insertion order, so ranking must not depend on iterating one.
      # Voters are listed rather than sampled: `scripted/1` hands out replies in
      # execution order, and concurrent voters make that a race.
      voters = [echoing("BETA"), echoing("ALPHA")]

      for _attempt <- 1..5 do
        assert {:ok, %{answer: "beta", agreement: 0.5}} = Consensus.run("q", voters: voters)
      end
    end
  end
end
