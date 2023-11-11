using Clang.Generators

cd(@__DIR__)

include_dir = normpath(@__DIR__, "..", "c_wrapper")

options = load_options(joinpath(@__DIR__, "generator.toml"))

args = get_default_args()
push!(args, "-I$include_dir")

headers = [joinpath(include_dir, "torch_api.h")]

ctx = create_context(headers, args, options)

build!(ctx, BUILDSTAGE_NO_PRINTING)

function rewrite!(e::Expr)
    if e.head == :function
        rewrite!(e, Val(e.head))
    end
end

function rewrite!(e::Expr, ::Val{:function})
    rewrite!(e.args[2], Val(e.args[2].head))
end

function rewrite!(e::Expr, ::Val{:block})
    e.args[1] = Expr(:macrocall, Symbol("@runtime_error_check"), nothing, e.args[1])
end

function rewrite!(dag::ExprDAG)
    for node in get_nodes(dag)
        for expr in get_exprs(node)
            rewrite!(expr)
        end
    end
end

rewrite!(ctx.dag)

build!(ctx, BUILDSTAGE_PRINTING_ONLY)
