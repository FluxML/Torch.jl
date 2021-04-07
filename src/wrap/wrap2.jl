using Clang

const LIBTORCH_INCLUDE = joinpath(PATH) |> normpath
const LIBTORCH_HEADERS = ["path/to/torch_api.h"] 

hs = ["path/to/torch_api.h",
      "path/to/torch_api_generated.h"]

@show LIBTORCH_INCLUDE
@show LIBTORCH_HEADERS

# create a work context
ctx = DefaultContext()

# parse headers
parse_headers!(ctx, LIBTORCH_HEADERS,
               args=["-I", LIBTORCH_INCLUDE],
               includes=vcat(LIBTORCH_INCLUDE, CLANG_INCLUDE),
               )

# settings
ctx.libname = ":libdoeye_caml"
ctx.options["is_function_strictly_typed"] = false
ctx.options["is_struct_mutable"] = false

# write output
api_file = joinpath(@__DIR__, "libdoeye_caml_generated.jl")
api_stream = open(api_file, "w")

ch = []

for trans_unit in ctx.trans_units
  root_cursor = getcursor(trans_unit)
  push!(ctx.cursor_stack, root_cursor)
  header = spelling(root_cursor)
  @info "wrapping header: $header ..."
  ctx.children = children(root_cursor)
  for (i, child) in enumerate(ctx.children)
    child_name = name(child)
    child_header = filename(child)
    ctx.children_index = i

    # TODO: Remove this `ch`
    push!(ch, child_header)

    startswith(child_name, "__") && continue  # skip compiler definitions
    child_name in keys(ctx.common_buffer) && continue  # already wrapped

    # child_header != header && continue
    !(child_header in hs) && continue  # skip if cursor filename is not in the headers to be wrapped

    wrap!(ctx, child)
  end

  @info "writing $(api_file)"
  println(api_stream, "# Julia wrapper for header: $(basename(header))")
  println(api_stream, "# Automatically generated using Clang.jl\n")
  print_buffer(api_stream, ctx.api_buffer)
  empty!(ctx.api_buffer)  # clean up api_buffer for the next header
end

close(api_stream)

common_file = joinpath(@__DIR__, "libtorch_common.jl")
open(common_file, "w") do f
    println(f, "# Automatically generated using Clang.jl\n")
    print_buffer(f, dump_to_buffer(ctx.common_buffer))
end

