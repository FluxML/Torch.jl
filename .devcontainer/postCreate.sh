sudo chown -R vscode:vscode /opt/juliaup /opt/julia_depot

CUDNN_ROOT=$(julia --project=$(mktemp -d) --eval '
        using Pkg;
        Pkg.add(name="CUDNN_jll", version=ENV["CUDNN_VERSION"]);
        using CUDNN_jll;
        println(CUDNN_jll.artifact_dir)') \
&& for F in $CUDNN_ROOT/include/cudnn*.h; do ln -s $F /usr/local/cuda/include/$(basename $F); done \
&& for F in $CUDNN_ROOT/lib/libcudnn*; do ln -s $F /usr/local/cuda/lib64/$(basename $F); done

opam init --disable-sandboxing --auto-setup
