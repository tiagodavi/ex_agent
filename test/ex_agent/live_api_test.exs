defmodule ExAgent.LiveApiTest do
  @moduledoc """
  End-to-end exercise of the library against **real provider APIs**.

  Every other test in this suite mocks HTTP. This one does not: it is the only
  place where request shapes, field names, size thresholds, and enum values are
  checked against what the providers actually accept.

  These tests are tagged `:external` and excluded by default — they cost money
  and need credentials. Run them deliberately:

      # everything you have credentials for
      mix test --only external

      # one provider
      mix test --only external test/ex_agent/live_api_test.exs --only openai

  ## Credentials

  A section whose credentials are missing is **skipped with a reason**, so a
  partial setup reports what it did not cover instead of silently omitting it.
  Test files are evaluated per run, so exporting a variable takes effect
  immediately.

      export OPENAI_API_KEY=sk-...
      export GEMINI_API_KEY=AIza...

      # OpenAI-compatible endpoint (vLLM / Modal / OpenRouter / Together / Groq)
      export EX_AGENT_COMPAT_BASE_URL=https://your-app.modal.run/v1
      export EX_AGENT_COMPAT_MODEL=Qwen/Qwen3-VL-8B-Instruct
      export EX_AGENT_COMPAT_API_KEY=...                 # optional, bearer auth
      export EX_AGENT_COMPAT_HEADERS='Modal-Key: k, Modal-Secret: s'  # optional
      export EX_AGENT_COMPAT_MODALITIES=text,image,video # optional, default text

      # embeddings usually live on a separate deployment; only then are they tested
      export EX_AGENT_COMPAT_EMBED_MODEL=jinaai/jina-embeddings-v3
      export EX_AGENT_COMPAT_EMBED_BASE_URL=https://your-embed-app.modal.run/v1  # optional
      export EX_AGENT_COMPAT_EMBED_API_KEY=...                                   # optional

  ## Optional extras

      # exercises URL attachments; must be publicly reachable by the provider.
      # Set the MIME var too when the URL has no file extension.
      export EX_AGENT_TEST_IMAGE_URL=https://.../photo.png
      export EX_AGENT_TEST_IMAGE_MIME=image/jpeg

      # exercises video on Gemini and on a video-capable compatible endpoint
      export EX_AGENT_TEST_VIDEO_PATH=/path/to/clip.mp4
      export EX_AGENT_TEST_VIDEO_URL=https://.../clip.mp4
  """

  use ExUnit.Case, async: false

  alias ExAgent.{Chunk, Embeddings, Error, Message, Provider, Response, Roles, Skill, Tool}
  alias ExAgent.Patterns.Subagents
  alias ExAgent.Providers.{Gemini, OpenAI, OpenAICompatible}

  @moduletag :external
  # Real models think slowly; a streamed reasoning turn can take minutes.
  @moduletag timeout: :timer.minutes(5)

  @openai_key System.get_env("OPENAI_API_KEY")
  @gemini_key System.get_env("GEMINI_API_KEY")

  @compat_base_url System.get_env("EX_AGENT_COMPAT_BASE_URL")
  @compat_model System.get_env("EX_AGENT_COMPAT_MODEL")
  @compat_key System.get_env("EX_AGENT_COMPAT_API_KEY")
  @compat_headers System.get_env("EX_AGENT_COMPAT_HEADERS")
  @compat_modalities System.get_env("EX_AGENT_COMPAT_MODALITIES")
  @compat_embed_model System.get_env("EX_AGENT_COMPAT_EMBED_MODEL")
  @compat_embed_base_url System.get_env("EX_AGENT_COMPAT_EMBED_BASE_URL")
  @compat_embed_key System.get_env("EX_AGENT_COMPAT_EMBED_API_KEY")

  @image_url System.get_env("EX_AGENT_TEST_IMAGE_URL")
  @image_mime System.get_env("EX_AGENT_TEST_IMAGE_MIME")
  @video_path System.get_env("EX_AGENT_TEST_VIDEO_PATH")
  @video_url System.get_env("EX_AGENT_TEST_VIDEO_URL")

  # Everything compiles regardless of which credentials are present; missing ones
  # turn into a `skip:` tag. That keeps one alias block valid at the top of the
  # file — compile-time `if` blocks would leave aliases unused in partial runs —
  # and makes uncovered sections visible in the run output.
  @skip_openai !@openai_key && "OPENAI_API_KEY is not set"
  @skip_gemini !@gemini_key && "GEMINI_API_KEY is not set"
  @skip_compat !(@compat_base_url && @compat_model) &&
                 "EX_AGENT_COMPAT_BASE_URL / EX_AGENT_COMPAT_MODEL are not set"
  @skip_chat !(@openai_key || @gemini_key) && "neither OPENAI_API_KEY nor GEMINI_API_KEY is set"

  @skip_image_url !@image_url && "EX_AGENT_TEST_IMAGE_URL is not set"
  @skip_video_path !@video_path && "EX_AGENT_TEST_VIDEO_PATH is not set"
  @skip_video_url !@video_url && "EX_AGENT_TEST_VIDEO_URL is not set"

  # `||` short-circuits while the module is being compiled, which would leave the
  # later attribute looking unread; taking the first truthy entry of a list reads
  # every one of them.
  @skip_compat_embed Enum.find(
                       [
                         @skip_compat,
                         !@compat_embed_model && "EX_AGENT_COMPAT_EMBED_MODEL is not set"
                       ],
                       & &1
                     ) || false
  @skip_openai_image_url Enum.find([@skip_openai, @skip_image_url], & &1) || false
  @skip_gemini_image_url Enum.find([@skip_gemini, @skip_image_url], & &1) || false
  @skip_gemini_video_path Enum.find([@skip_gemini, @skip_video_path], & &1) || false
  @skip_compat_image_url Enum.find([@skip_compat, @skip_image_url], & &1) || false
  @skip_compat_video_path Enum.find([@skip_compat, @skip_video_path], & &1) || false
  @skip_compat_video_url Enum.find([@skip_compat, @skip_video_url], & &1) || false

  # A real 1x1 red PNG. Providers reject bytes that are not decodable images,
  # so this cannot be a placeholder string.
  @png Base.decode64!(
         "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
       )

  @pdf """
  %PDF-1.4
  1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
  2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
  3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 99 9]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
  4 0 obj<</Length 46>>stream
  BT /F1 8 Tf 4 2 Td (MAGIC WORD: rutabaga) Tj ET
  endstream
  endobj
  5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
  trailer<</Root 1 0 R>>
  """

  setup_all do
    dir = Path.join(System.tmp_dir!(), "ex_agent_live_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf(dir) end)

    png_path = Path.join(dir, "dot.png")
    File.write!(png_path, @png)

    pdf_path = Path.join(dir, "doc.pdf")
    File.write!(pdf_path, @pdf)

    %{dir: dir, png_path: png_path, pdf_path: pdf_path}
  end

  # --- Shared helpers ---------------------------------------------------------

  defp weather_tool do
    {:ok, tool} =
      Tool.new(
        name: "get_weather",
        description: "Get the current weather for a city.",
        parameters: %{
          "type" => "object",
          "properties" => %{"city" => %{"type" => "string"}},
          "required" => ["city"]
        },
        function: fn %{"city" => city} -> "#{city}: 22C and sunny" end
      )

    tool
  end

  defp start_agent!(provider, opts \\ []) do
    {:ok, pid} = ExAgent.start_agent([provider: provider] ++ opts)
    on_exit(fn -> if Process.alive?(pid), do: ExAgent.stop_agent(pid) end)
    pid
  end

  defp texts(chunks),
    do: chunks |> Enum.filter(&(&1.type == :text_delta)) |> Enum.map(& &1.text)

  # `EX_AGENT_TEST_IMAGE_MIME` is only needed when the URL carries no extension —
  # ExAgent never fetches a URL to sniff its type.
  defp image_file(extra \\ %{}) do
    base =
      if @image_mime, do: %{url: @image_url, mime_type: @image_mime}, else: %{url: @image_url}

    Map.merge(base, extra)
  end

  # Models phrase things freely; assert on substance, not exact wording.
  defp assert_mentions(content, expected) do
    assert content =~ ~r/#{expected}/i,
           "expected the response to mention #{inspect(expected)}, got: #{inspect(content)}"
  end

  defp openai(opts \\ []), do: OpenAI.new(Keyword.merge([api_key: @openai_key], opts))
  defp gemini(opts \\ []), do: Gemini.new(Keyword.merge([api_key: @gemini_key], opts))

  defp parse_headers(raw) do
    case raw do
      nil ->
        []

      raw ->
        raw
        |> String.split(",", trim: true)
        |> Enum.map(fn pair ->
          [name, value] = String.split(pair, ":", parts: 2)
          {String.trim(name), String.trim(value)}
        end)
    end
  end

  defp parse_modalities(raw) do
    case raw do
      nil -> [:text]
      raw -> raw |> String.split(",", trim: true) |> Enum.map(&String.to_atom(String.trim(&1)))
    end
  end

  # Merged, not appended: a duplicate key reaches NimbleOptions as an
  # "unknown options [:modalities]" error that also lists :modalities as valid.
  defp compat(opts \\ []) do
    OpenAICompatible.new(
      Keyword.merge(
        [
          base_url: @compat_base_url,
          model: @compat_model,
          api_key: @compat_key,
          headers: parse_headers(@compat_headers),
          modalities: parse_modalities(@compat_modalities)
        ],
        opts
      )
    )
  end

  defp compat_embed do
    OpenAICompatible.new(
      base_url: @compat_embed_base_url || @compat_base_url,
      model: @compat_embed_model,
      api_key: @compat_embed_key || @compat_key,
      headers: parse_headers(@compat_headers)
    )
  end

  # The multi-agent patterns are provider-agnostic; they run on whichever chat
  # credential is present. The last clause keeps them compiling with none — those
  # tests carry @skip_chat.
  cond do
    @openai_key ->
      defp pattern_provider(opts \\ []), do: openai(opts)
      defp role_spec, do: {OpenAI, api_key: @openai_key}

    @gemini_key ->
      defp pattern_provider(opts \\ []), do: gemini(opts)
      defp role_spec, do: {Gemini, api_key: @gemini_key}

    true ->
      # Unreachable at runtime — every caller carries @skip_chat — but it has to
      # type-check, so it delegates rather than raising.
      defp pattern_provider(opts \\ []), do: openai(opts)
      defp role_spec, do: {OpenAI, api_key: @openai_key}
  end

  # ============================================================================
  # OpenAI
  # ============================================================================

  describe "OpenAI · chat" do
    @describetag :openai
    @describetag skip: @skip_openai

    test "returns a Response carrying content, usage and finish reason" do
      agent = start_agent!(openai())

      assert {:ok, %Response{} = response} = ExAgent.chat(agent, "Reply with exactly: pong")

      assert_mentions(response.content, "pong")
      assert response.finish_reason == :stop
      assert response.usage.total_tokens > 0
      assert %Message{role: :assistant} = response.message
    end

    test "honours a system prompt" do
      agent = start_agent!(openai(system_prompt: "Always answer in French."))

      assert {:ok, response} = ExAgent.chat(agent, "What colour is the sky?")
      assert_mentions(response.content, "ciel|bleu")
    end

    test "keeps conversation context across turns" do
      agent = start_agent!(openai())

      assert {:ok, _} = ExAgent.chat(agent, "My favourite number is 41. Acknowledge briefly.")
      assert {:ok, response} = ExAgent.chat(agent, "Add one to my favourite number.")

      assert_mentions(response.content, "42")
      assert length(ExAgent.get_context(agent).messages) == 4
    end
  end

  describe "OpenAI · tool calling" do
    @describetag :openai
    @describetag skip: @skip_openai

    test "executes a tool and folds the result into the answer" do
      agent = start_agent!(openai(), tools: [weather_tool()])

      assert {:ok, response} = ExAgent.chat(agent, "What is the weather in Lisbon?")

      assert_mentions(response.content, "22")

      # user, assistant tool-call, tool result, final assistant
      assert length(ExAgent.get_context(agent).messages) >= 4
    end
  end

  describe "OpenAI · streaming" do
    @describetag :openai
    @describetag skip: @skip_openai

    test "yields text chunks and terminates with a single done chunk" do
      agent = start_agent!(openai())

      chunks = agent |> ExAgent.chat_stream("Count from 1 to 5, digits only.") |> Enum.to_list()

      assert Enum.count(chunks, &(&1.type == :done)) == 1
      assert %Chunk{type: :done, finish_reason: :stop} = List.last(chunks)
      assert_mentions(Enum.join(texts(chunks)), "5")
    end

    test "collect/1 reproduces the blocking response shape" do
      agent = start_agent!(openai())

      assert {:ok, %Response{} = response} =
               agent |> ExAgent.chat_stream("Reply with exactly: pong") |> ExAgent.collect()

      assert_mentions(response.content, "pong")
      assert response.finish_reason == :stop
      assert response.usage.total_tokens > 0
    end

    test "commits the streamed turn to context" do
      agent = start_agent!(openai())

      agent |> ExAgent.chat_stream("Say hello.") |> Stream.run()

      assert length(ExAgent.get_context(agent).messages) == 2
      assert :sys.get_state(agent).status == :idle
    end
  end

  describe "OpenAI · attachments" do
    @describetag :openai
    @describetag skip: @skip_openai

    test "reads an inline image from raw bytes" do
      agent = start_agent!(openai())

      assert {:ok, response} =
               ExAgent.chat(agent, "What colour is this 1x1 image? One word.",
                 files: [%{data: @png, mime_type: "image/png"}]
               )

      assert is_binary(response.content) and response.content != ""
    end

    test "reads an image from a path, inferring the mime type", %{png_path: path} do
      agent = start_agent!(openai())

      assert {:ok, response} =
               ExAgent.chat(agent, "Describe this image in one word.", files: [%{path: path}])

      assert is_binary(response.content) and response.content != ""
    end

    test "reads a document from a path", %{pdf_path: path} do
      agent = start_agent!(openai())

      assert {:ok, response} =
               ExAgent.chat(agent, "What is the magic word in this PDF?", files: [%{path: path}])

      assert is_binary(response.content) and response.content != ""
    end

    @tag skip: @skip_openai_image_url
    test "references an image by URL without downloading it" do
      agent = start_agent!(openai())

      assert {:ok, response} =
               ExAgent.chat(agent, "Describe this image in one sentence.", files: [image_file()])

      assert is_binary(response.content) and response.content != ""
    end
  end

  describe "OpenAI · file uploads" do
    @describetag :openai
    @describetag skip: @skip_openai

    test "uploads a file and references it by id", %{pdf_path: path} do
      provider = openai()

      assert {:ok, ref} = ExAgent.upload_file(provider, path, "application/pdf")
      assert ref.provider == :openai
      assert is_binary(ref.file_id)

      agent = start_agent!(provider)

      assert {:ok, response} =
               ExAgent.chat(agent, "Summarize the attached file in one sentence.",
                 files: [%{file_ref: ref}]
               )

      assert is_binary(response.content) and response.content != ""
    end

    test "uploads raw binary data" do
      assert {:ok, ref} = ExAgent.upload_data(openai(), @png, "image/png", filename: "dot.png")

      assert is_binary(ref.file_id)
    end
  end

  describe "OpenAI · embeddings" do
    @describetag :openai
    @describetag skip: @skip_openai

    test "embeds a single string" do
      assert {:ok, %Embeddings{} = result} = ExAgent.embed(openai(), "hello world")

      assert [vector] = result.vectors
      assert length(vector) == result.dimensions
      assert result.model == "text-embedding-3-small"
      assert result.usage.total_tokens > 0
    end

    test "embeds a batch, one vector per input, in order" do
      assert {:ok, result} = ExAgent.embed(openai(), ["alpha", "beta", "gamma"])

      assert length(result.vectors) == 3
      assert Enum.all?(result.vectors, &(length(&1) == result.dimensions))
    end

    test "truncates to the requested dimensions" do
      assert {:ok, result} = ExAgent.embed(openai(), ["hello"], dimensions: 256)

      assert result.dimensions == 256
      assert [vector] = result.vectors
      assert length(vector) == 256
    end

    test "rejects a task type rather than silently dropping it" do
      assert {:error, %Error{type: :unsupported}} =
               ExAgent.embed(openai(), ["hello"], task: :retrieval_query)
    end

    test "produces vectors where related text scores above unrelated text" do
      assert {:ok, result} =
               ExAgent.embed(openai(), [
                 "The cat sat on the mat.",
                 "A feline rested on the rug.",
                 "Quarterly revenue grew by twelve percent."
               ])

      [cat, feline, revenue] = result.vectors

      assert Embeddings.cosine_similarity(cat, feline) >
               Embeddings.cosine_similarity(cat, revenue)
    end
  end

  describe "OpenAI · built-in tools" do
    @describetag :openai
    @describetag skip: @skip_openai

    test "accepts the web_search built-in tool" do
      # web_search_options needs a *-search-preview model, and those reject
      # `temperature` — so the provider default has to be turned off.
      agent = start_agent!(openai(model: "gpt-4o-search-preview", temperature: nil))

      assert {:ok, response} =
               ExAgent.chat(agent, "What is the capital of Portugal?",
                 built_in_tools: [:web_search]
               )

      assert_mentions(response.content, "lisbon|lisboa")
    end
  end

  describe "OpenAI · errors" do
    @describetag :openai
    @describetag skip: @skip_openai

    test "an invalid key returns a non-retryable auth error" do
      agent = start_agent!(OpenAI.new(api_key: "sk-definitely-not-a-real-key"))

      assert {:error, %Error{type: :auth} = error} = ExAgent.chat(agent, "Hi")
      refute error.retryable?
      assert error.status in [401, 403]
    end

    test "an unknown model returns an actionable error" do
      agent = start_agent!(openai(model: "gpt-does-not-exist"))

      assert {:error, %Error{} = error} = ExAgent.chat(agent, "Hi")
      assert error.type in [:not_found, :invalid_request]
      refute error.retryable?
    end
  end

  # ============================================================================
  # Gemini
  # ============================================================================

  describe "Gemini · chat" do
    @describetag :gemini
    @describetag skip: @skip_gemini

    test "returns a Response carrying content, usage and finish reason" do
      agent = start_agent!(gemini())

      assert {:ok, %Response{} = response} = ExAgent.chat(agent, "Reply with exactly: pong")

      assert_mentions(response.content, "pong")
      assert response.finish_reason == :stop
      assert response.usage.total_tokens > 0
    end

    test "honours a system prompt" do
      agent = start_agent!(gemini(system_prompt: "Always answer in French."))

      assert {:ok, response} = ExAgent.chat(agent, "What colour is the sky?")
      assert_mentions(response.content, "ciel|bleu")
    end
  end

  describe "Gemini · tool calling" do
    @describetag :gemini
    @describetag skip: @skip_gemini

    test "executes a tool and folds the result into the answer" do
      agent = start_agent!(gemini(), tools: [weather_tool()])

      assert {:ok, response} = ExAgent.chat(agent, "What is the weather in Lisbon?")
      assert_mentions(response.content, "22")
    end
  end

  describe "Gemini · streaming" do
    @describetag :gemini
    @describetag skip: @skip_gemini

    test "yields text chunks and terminates with a single done chunk" do
      agent = start_agent!(gemini())

      chunks = agent |> ExAgent.chat_stream("Count from 1 to 5, digits only.") |> Enum.to_list()

      assert Enum.count(chunks, &(&1.type == :done)) == 1
      assert_mentions(Enum.join(texts(chunks)), "5")
    end

    test "collect/1 reproduces the blocking response shape" do
      agent = start_agent!(gemini())

      assert {:ok, %Response{} = response} =
               agent |> ExAgent.chat_stream("Reply with exactly: pong") |> ExAgent.collect()

      assert_mentions(response.content, "pong")
    end
  end

  describe "Gemini · attachments" do
    @describetag :gemini
    @describetag skip: @skip_gemini

    test "reads an inline image from raw bytes" do
      agent = start_agent!(gemini())

      assert {:ok, response} =
               ExAgent.chat(agent, "What colour is this 1x1 image? One word.",
                 files: [%{data: @png, mime_type: "image/png"}]
               )

      assert is_binary(response.content) and response.content != ""
    end

    test "reads a PDF from a path", %{pdf_path: path} do
      agent = start_agent!(gemini())

      assert {:ok, response} =
               ExAgent.chat(agent, "What is the magic word in this PDF?", files: [%{path: path}])

      assert is_binary(response.content) and response.content != ""
    end

    @tag skip: @skip_gemini_image_url
    test "references an image by URL without downloading it" do
      agent = start_agent!(gemini())

      assert {:ok, response} =
               ExAgent.chat(agent, "Describe this image in one sentence.", files: [image_file()])

      assert is_binary(response.content) and response.content != ""
    end

    @tag skip: @skip_gemini_video_path
    test "uploads a video, waits for it to become ACTIVE, and reads it" do
      agent = start_agent!(gemini())

      assert {:ok, response} =
               ExAgent.chat(agent, "Describe this video in one sentence.",
                 files: [%{path: @video_path, fps: 1}]
               )

      assert is_binary(response.content) and response.content != ""
    end
  end

  describe "Gemini · file uploads" do
    @describetag :gemini
    @describetag skip: @skip_gemini

    test "uploads a file and references it by URI", %{pdf_path: path} do
      provider = gemini()

      assert {:ok, ref} = ExAgent.upload_file(provider, path, "application/pdf")
      assert ref.provider == :gemini
      assert is_binary(ref.file_uri)

      agent = start_agent!(provider)

      assert {:ok, response} =
               ExAgent.chat(agent, "Summarize the attached file in one sentence.",
                 files: [%{file_ref: ref}]
               )

      assert is_binary(response.content) and response.content != ""
    end
  end

  describe "Gemini · embeddings (gemini-embedding-001, taskType family)" do
    @describetag :gemini
    @describetag skip: @skip_gemini

    test "embeds a batch, one vector per input" do
      assert {:ok, %Embeddings{} = result} =
               ExAgent.embed(gemini(), ["alpha", "beta", "gamma"], task: :retrieval_document)

      assert length(result.vectors) == 3
      assert result.model == "gemini-embedding-001"
      assert result.task == :retrieval_document
    end

    test "accepts every normalized task" do
      for task <- Embeddings.tasks() do
        assert {:ok, %Embeddings{task: ^task}} = ExAgent.embed(gemini(), ["hello"], task: task),
               "task #{inspect(task)} was rejected"
      end
    end

    test "returns unit-length vectors when dimensions are truncated" do
      assert {:ok, result} =
               ExAgent.embed(gemini(), ["hello"], dimensions: 768, task: :similarity)

      assert [vector] = result.vectors
      assert length(vector) == 768

      magnitude = vector |> Enum.reduce(0.0, fn v, acc -> acc + v * v end) |> :math.sqrt()
      assert_in_delta magnitude, 1.0, 1.0e-4
    end

    test "asymmetric retrieval ranks the right document first" do
      assert {:ok, docs} =
               ExAgent.embed(
                 gemini(),
                 ["OTP supervisors restart crashed processes.", "Espresso needs fine grounds."],
                 task: :retrieval_document
               )

      assert {:ok, query} =
               ExAgent.embed(gemini(), "what restarts failed processes?", task: :retrieval_query)

      [supervisors, coffee] = docs.vectors
      [q] = query.vectors

      assert Embeddings.cosine_similarity(q, supervisors) >
               Embeddings.cosine_similarity(q, coffee)
    end

    test "rejects an unknown embedding model instead of guessing a family" do
      assert {:error, %Error{type: :unsupported} = error} =
               ExAgent.embed(gemini(), ["hello"], model: "gemini-embedding-99")

      assert error.message =~ "embedding_family"
    end
  end

  describe "Gemini · embeddings (gemini-embedding-2, prefix family)" do
    @describetag :gemini
    @describetag :gemini_embedding_2
    @describetag skip: @skip_gemini

    # The regression that matters most: a flat list returns ONE aggregated
    # vector on this model unless each input gets its own Content.
    test "three inputs return three vectors, not one" do
      assert {:ok, result} =
               ExAgent.embed(gemini(), ["one", "two", "three"],
                 model: "gemini-embedding-2",
                 task: :retrieval_document
               )

      assert length(result.vectors) == 3
    end

    test "accepts a title for the document template" do
      assert {:ok, result} =
               ExAgent.embed(gemini(), [%{content: "body text", title: "OTP Guide"}],
                 model: "gemini-embedding-2",
                 task: :retrieval_document
               )

      assert [_vector] = result.vectors
    end
  end

  describe "Gemini · built-in tools" do
    @describetag :gemini
    @describetag skip: @skip_gemini

    test "accepts the google_search built-in tool" do
      agent = start_agent!(gemini())

      assert {:ok, response} =
               ExAgent.chat(agent, "What is the capital of Portugal?",
                 built_in_tools: [:google_search]
               )

      assert_mentions(response.content, "lisbon|lisboa")
    end
  end

  describe "Gemini · errors" do
    @describetag :gemini
    @describetag skip: @skip_gemini

    test "an invalid key returns a non-retryable auth error" do
      agent = start_agent!(Gemini.new(api_key: "AIza-definitely-not-real"))

      assert {:error, %Error{} = error} = ExAgent.chat(agent, "Hi")
      assert error.type in [:auth, :invalid_request]
      refute error.retryable?
    end
  end

  # ============================================================================
  # OpenAI-compatible (vLLM / Modal / OpenRouter / Together / Groq)
  # ============================================================================

  describe "OpenAICompatible · configuration" do
    @describetag :compat
    @describetag skip: @skip_compat

    test "probe/1 confirms the endpoint serves the configured model" do
      case OpenAICompatible.probe(compat()) do
        :ok ->
          :ok

        {:error, %Error{type: :not_found} = error} ->
          flunk("endpoint does not serve #{@compat_model}: #{error.message}")

        {:error, %Error{} = error} ->
          flunk("probe failed: #{Exception.message(error)}")
      end
    end
  end

  describe "OpenAICompatible · chat" do
    @describetag :compat
    @describetag skip: @skip_compat

    test "returns a Response carrying content" do
      agent = start_agent!(compat())

      assert {:ok, %Response{} = response} = ExAgent.chat(agent, "Reply with exactly: pong")
      assert_mentions(response.content, "pong")
    end

    test "custom auth headers reach the endpoint" do
      # A 401/403 here means the Modal-Key / bearer plumbing is wrong.
      agent = start_agent!(compat())

      assert {:ok, _} = ExAgent.chat(agent, "Hi")
    end
  end

  describe "OpenAICompatible · tool calling" do
    @describetag :compat
    @describetag skip: @skip_compat

    test "executes a tool, or the endpoint reports tool calling is not enabled" do
      # vLLM only serves function calling when the container was started with
      # --enable-auto-tool-choice and --tool-call-parser. That is a deployment
      # property, like :modalities, so accept that one refusal — and nothing
      # else, so a malformed request or an auth failure still fails here.
      agent = start_agent!(compat(), tools: [weather_tool()])

      case ExAgent.chat(agent, "What is the weather in Lisbon?") do
        {:ok, response} ->
          assert_mentions(response.content, "22")

        {:error, %Error{type: :invalid_request, status: 400} = error} ->
          assert error.message =~ "tool"
      end
    end
  end

  describe "OpenAICompatible · streaming" do
    @describetag :compat
    @describetag skip: @skip_compat

    test "yields text chunks and terminates with a single done chunk" do
      agent = start_agent!(compat())

      chunks = agent |> ExAgent.chat_stream("Count from 1 to 5, digits only.") |> Enum.to_list()

      assert Enum.count(chunks, &(&1.type == :done)) == 1
      assert_mentions(Enum.join(texts(chunks)), "5")
    end

    test "streams a turn that carries an image attachment" do
      # Attachment preparation happens before the stream is built, so a broken
      # media part surfaces as a terminal error chunk rather than text.
      provider = compat()

      if :image in provider.modalities do
        agent = start_agent!(provider)

        chunks =
          agent
          |> ExAgent.chat_stream("Describe this image in one short sentence.",
            files: [%{data: @png, mime_type: "image/png"}]
          )
          |> Enum.to_list()

        assert Enum.count(chunks, &(&1.type == :done)) == 1
        assert List.last(chunks).error == nil
        assert texts(chunks) |> Enum.join() |> String.trim() != ""
      end
    end
  end

  describe "OpenAICompatible · attachments" do
    @describetag :compat
    @describetag skip: @skip_compat

    test "an image is sent as a data URI when the deployment declares :image" do
      provider = compat()

      if :image in provider.modalities do
        agent = start_agent!(provider)

        assert {:ok, response} =
                 ExAgent.chat(agent, "What colour is this 1x1 image? One word.",
                   files: [%{data: @png, mime_type: "image/png"}]
                 )

        assert is_binary(response.content) and response.content != ""
      end
    end

    @tag skip: @skip_compat_image_url
    test "an image URL is passed through rather than fetched" do
      provider = compat()

      if :image in provider.modalities do
        agent = start_agent!(provider)

        assert {:ok, response} =
                 ExAgent.chat(agent, "Describe this image in one short sentence.",
                   files: [image_file()]
                 )

        assert is_binary(response.content) and response.content != ""
      end
    end

    @tag skip: @skip_compat_image_url
    test "two attachments in one turn become two content parts" do
      # Mixes an inline data URI with a URL so both part shapes are assembled
      # into a single message, which is where a list-vs-single bug would show.
      provider = compat()

      if :image in provider.modalities do
        agent = start_agent!(provider)

        assert {:ok, response} =
                 ExAgent.chat(agent, "How many images did I send? Reply with a digit.",
                   files: [%{data: @png, mime_type: "image/png"}, image_file()]
                 )

        assert is_binary(response.content) and response.content != ""
      end
    end

    @tag skip: @skip_compat_image_url
    test "an extensionless URL without :mime_type fails loudly" do
      agent = start_agent!(compat())

      assert {:error, %Error{type: :invalid_request} = error} =
               ExAgent.chat(agent, "Describe this.", files: [%{url: "https://cdn.test/i/237"}])

      assert error.message =~ ":mime_type"
    end

    @tag skip: @skip_compat_video_path
    test "a video is inlined as a data URI from a local path" do
      provider = compat()

      if :video in provider.modalities do
        agent = start_agent!(provider)

        assert {:ok, response} =
                 ExAgent.chat(agent, "Describe this video in one sentence.",
                   files: [%{path: @video_path}]
                 )

        assert is_binary(response.content) and response.content != ""
      end
    end

    @tag skip: @skip_compat_video_url
    test "a video URL is passed through, and :fps rides along" do
      provider = compat()

      if :video in provider.modalities do
        agent = start_agent!(provider)

        assert {:ok, response} =
                 ExAgent.chat(agent, "Describe this video in one sentence.",
                   files: [%{url: @video_url, fps: 1}]
                 )

        assert is_binary(response.content) and response.content != ""
      end
    end

    test "an undeclared modality is rejected before any request" do
      provider = compat(modalities: [:text])

      agent = start_agent!(provider)

      assert {:error, %Error{type: :unsupported}} =
               ExAgent.chat(agent, "Describe this.",
                 files: [%{data: @png, mime_type: "image/png"}]
               )
    end

    test "an attachment past max_inline_bytes is refused with a clear message" do
      agent = start_agent!(compat(modalities: [:text, :image], max_inline_bytes: 10))

      assert {:error, %Error{type: :unsupported} = error} =
               ExAgent.chat(agent, "Describe this.",
                 files: [%{data: @png, mime_type: "image/png"}]
               )

      assert error.message =~ "max_inline_bytes"
    end
  end

  # A chat container and an embedding container are usually separate deployments,
  # so this needs its own model (and often its own URL and credential) rather than
  # assuming the chat endpoint serves both.
  describe "OpenAICompatible · embeddings" do
    @describetag :compat
    @describetag skip: @skip_compat_embed

    test "embeds a batch, one vector per input" do
      assert {:ok, %Embeddings{} = result} = ExAgent.embed(compat_embed(), ["alpha", "beta"])

      assert length(result.vectors) == 2
      assert result.model == @compat_embed_model
    end

    test "a task is translated into the endpoint's own vocabulary" do
      assert {:ok, %Embeddings{task: :retrieval_document}} =
               ExAgent.embed(compat_embed(), ["alpha"], task: :retrieval_document)
    end
  end

  # ============================================================================
  # Provider roles — config resolution is unit-tested; what only a live run
  # proves is that a role-resolved struct and the stateless `*_with` wrappers
  # build requests a real endpoint accepts, since they bypass the agent that
  # every other path goes through.
  # ============================================================================

  describe "Roles · resolution and one-shot wrappers" do
    @describetag :roles
    @describetag skip: @skip_chat

    setup do
      Application.put_env(:ex_agent, :roles, chat: role_spec())
      Roles.build!()

      on_exit(fn ->
        Application.delete_env(:ex_agent, :roles)
        Roles.build!()
      end)
    end

    test "a role-resolved struct drives a real agent" do
      agent = start_agent!(ExAgent.provider!(:chat))

      assert {:ok, response} = ExAgent.chat(agent, "Reply with exactly: pong")
      assert_mentions(response.content, "pong")
    end

    test "start_agent/1 accepts the role directly" do
      {:ok, agent} = ExAgent.start_agent(role: :chat)
      on_exit(fn -> if Process.alive?(agent), do: ExAgent.stop_agent(agent) end)

      assert {:ok, response} = ExAgent.chat(agent, "Reply with exactly: pong")
      assert_mentions(response.content, "pong")
    end

    test "chat_with/3 completes a turn without an agent process" do
      assert {:ok, response} =
               ExAgent.chat_with(:chat, "What is the capital of France? One word.")

      assert_mentions(response.content, "paris")
      assert response.usage.total_tokens > 0
    end

    test "chat_with/3 honours the provider's system prompt" do
      # The system prompt comes off the struct rather than from the agent, so
      # this is the path that would silently drop it.
      {module, opts} = role_spec()

      Application.put_env(:ex_agent, :roles,
        french: {module, opts ++ [system_prompt: "Always answer in French."]}
      )

      Roles.build!()

      assert {:ok, response} = ExAgent.chat_with(:french, "What is the capital of France?")
      assert_mentions(response.content, "capitale|est Paris")
    end

    test "stream_with/3 streams and collects into the same response shape" do
      chunks = :chat |> ExAgent.stream_with("Count from 1 to 5.") |> Enum.to_list()

      assert texts(chunks) != []
      assert {:ok, response} = ExAgent.collect(chunks)
      assert_mentions(response.content, "5")
    end

    test "provider!/2 overrides reach the wire without touching the cached struct" do
      overridden = ExAgent.provider!(:chat, max_tokens: 64)
      {:ok, message} = Message.new(role: :user, content: "Say hi.")

      assert overridden.max_tokens == 64
      assert ExAgent.provider!(:chat).max_tokens != 64
      assert {:ok, _response} = Provider.chat(overridden, [message])
    end
  end

  describe "Roles · embed_with/3" do
    @describetag :roles
    @describetag skip: @skip_chat

    setup do
      Application.put_env(:ex_agent, :roles, embed: role_spec())
      Roles.build!()

      on_exit(fn ->
        Application.delete_env(:ex_agent, :roles)
        Roles.build!()
      end)
    end

    test "a role routes an embedding request to the right endpoint" do
      assert {:ok, %Embeddings{vectors: [vector]}} =
               ExAgent.embed_with(:embed, "Elixir runs on the BEAM")

      assert length(vector) > 100
    end
  end

  # ============================================================================
  # Multi-agent patterns — provider-agnostic
  # ============================================================================

  describe "Patterns · subagents" do
    @describetag :patterns
    @describetag skip: @skip_chat

    test "an orchestrator delegates to a named subagent" do
      specs = [
        %{
          name: "mathematician",
          provider: pattern_provider(),
          system_prompt: "You answer arithmetic with digits only.",
          description: "Solves arithmetic problems"
        }
      ]

      assert {:ok, answer} = Subagents.invoke_subagent(hd(specs), "What is 6 * 7?")

      assert_mentions(answer, "42")
    end

    test "subagents run in parallel and all return" do
      specs_with_inputs = [
        {%{name: "a", provider: pattern_provider(), description: "first"}, "Reply: OK"},
        {%{name: "b", provider: pattern_provider(), description: "second"}, "Reply: OK"}
      ]

      results = Subagents.invoke_subagents_parallel(specs_with_inputs)

      assert length(results) == 2
      assert Enum.all?(results, fn {_name, result} -> match?({:ok, _}, result) end)
    end

    test "an orchestrator agent can invoke subagents as tools" do
      specs = [
        %{
          name: "mathematician",
          description: "Solves arithmetic problems",
          provider: pattern_provider(),
          system_prompt: "You answer arithmetic with digits only."
        }
      ]

      tools = Subagents.build_orchestrator_tools(specs)
      agent = start_agent!(pattern_provider(), tools: tools)

      assert {:ok, response} = ExAgent.chat(agent, "Ask the mathematician what 6 * 7 is.")

      assert_mentions(response.content, "42")
    end
  end

  describe "Patterns · router" do
    @describetag :patterns
    @describetag skip: @skip_chat

    test "dispatches to matching agents in parallel and synthesizes one answer" do
      biology = start_agent!(pattern_provider(system_prompt: "You are a biologist."))
      history = start_agent!(pattern_provider(system_prompt: "You are a historian."))

      routes = [
        %{
          name: "biology",
          agent: biology,
          match_fn: fn input -> String.contains?(String.downcase(input), "selection") end
        },
        %{
          name: "history",
          agent: history,
          match_fn: fn input -> String.contains?(String.downcase(input), "described") end
        }
      ]

      assert {:ok, answer} =
               ExAgent.route("Who first described natural selection?",
                 routes: routes,
                 timeout: 120_000
               )

      assert_mentions(answer, "darwin")
    end

    test "a custom synthesizer receives every agent result" do
      agent = start_agent!(pattern_provider())

      routes = [%{name: "only", agent: agent, match_fn: fn _input -> true end}]

      assert {:ok, answer} =
               ExAgent.route("Reply with exactly: pong",
                 routes: routes,
                 timeout: 120_000,
                 synthesizer: fn _input, results ->
                   assert [{"only", content}] = results
                   "synthesized:" <> content
                 end
               )

      assert String.starts_with?(answer, "synthesized:")
    end

    test "input matching no route reports it rather than guessing" do
      agent = start_agent!(pattern_provider())
      routes = [%{name: "never", agent: agent, match_fn: fn _input -> false end}]

      assert {:error, :no_matching_routes} = ExAgent.route("anything", routes: routes)
    end
  end

  describe "Patterns · skills" do
    @describetag :patterns
    @describetag skip: @skip_chat

    test "a skill activates and changes the agent's behaviour" do
      {:ok, pirate} =
        Skill.new(
          name: "pirate",
          system_prompt: "You are a pirate. Always say 'arrr' in your reply.",
          activation_fn: fn context ->
            Enum.any?(context.messages, &String.contains?(&1.content, "treasure"))
          end
        )

      agent = start_agent!(pattern_provider(), skills: [pirate])

      assert {:ok, response} = ExAgent.chat(agent, "Where is the treasure?")
      assert_mentions(response.content, "arr")
    end
  end

  describe "Patterns · handoff" do
    @describetag :patterns
    @describetag skip: @skip_chat

    test "context transfers to the target agent" do
      source = start_agent!(pattern_provider())
      target = start_agent!(pattern_provider())

      assert {:ok, _} = ExAgent.chat(source, "My name is Ada. Acknowledge briefly.")

      :ok = ExAgent.handoff(target, ExAgent.get_context(source))

      assert {:ok, response} = ExAgent.chat(target, "What is my name?")
      assert_mentions(response.content, "ada")
    end
  end

  describe "Concurrency" do
    @describetag :patterns
    @describetag skip: @skip_chat

    test "chat_async/3 returns a Task that resolves to a Response" do
      agent = start_agent!(pattern_provider())

      assert {:ok, %Response{}} =
               agent |> ExAgent.chat_async("Reply with exactly: pong") |> Task.await(60_000)
    end

    test "an agent is busy while a request is in flight" do
      agent = start_agent!(pattern_provider())

      task = ExAgent.chat_async(agent, "Write one paragraph about OTP.")
      # Give the first request time to reach :processing.
      Process.sleep(200)

      assert {:error, :busy} = ExAgent.chat(agent, "Interrupt")
      assert {:ok, _} = Task.await(task, 60_000)
    end

    test "reset/1 clears conversation context" do
      agent = start_agent!(pattern_provider())

      assert {:ok, _} = ExAgent.chat(agent, "Remember the number 7.")
      assert ExAgent.get_context(agent).messages != []

      :ok = ExAgent.reset(agent)
      assert ExAgent.get_context(agent).messages == []
    end
  end
end
