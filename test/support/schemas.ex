defmodule ExAgent.Test.Schemas do
  @moduledoc false

  # Ordinary Ecto embedded schemas, deliberately written with no reference to
  # ExAgent - that is the contract `ExAgent.Schema` has to satisfy.

  defmodule Line do
    @moduledoc false
    use Ecto.Schema

    @primary_key false
    embedded_schema do
      field :description, :string
      field :amount, :float
    end
  end

  defmodule Invoice do
    @moduledoc false
    use Ecto.Schema

    @primary_key false
    embedded_schema do
      field :total, :float
      field :currency, Ecto.Enum, values: [:EUR, :USD, :GBP]
      field :issued_on, :date
      embeds_many :lines, Line
    end
  end

  # Keeps Ecto's default binary_id primary key, to prove it is excluded rather
  # than refused.
  defmodule WithPrimaryKey do
    @moduledoc false
    use Ecto.Schema

    embedded_schema do
      field :name, :string
    end
  end

  defmodule Address do
    @moduledoc false
    use Ecto.Schema

    @primary_key false
    embedded_schema do
      field :city, :string
    end
  end

  defmodule Person do
    @moduledoc false
    use Ecto.Schema

    @primary_key false
    embedded_schema do
      field :name, :string
      field :age, :integer
      field :active, :boolean
      field :tags, {:array, :string}
      field :seen_at, :utc_datetime
      embeds_one :address, Address
    end
  end

  # `:map` has no expressible JSON Schema under OpenAI strict mode.
  defmodule FreeMap do
    @moduledoc false
    use Ecto.Schema

    @primary_key false
    embedded_schema do
      field :payload, :map
    end
  end

  defmodule Empty do
    @moduledoc false
    use Ecto.Schema

    @primary_key false
    embedded_schema do
    end
  end
end
