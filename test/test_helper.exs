# `:external` tests hit real provider APIs, cost money, and need credentials, so
# they are excluded unless explicitly requested:
#
#     mix test --include external
#     mix test --only external
ExUnit.start(exclude: [:external])
