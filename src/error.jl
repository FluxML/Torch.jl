macro runtime_error_check(ex)
  quote
    x = $ex
    if x == 1
      cs = cglobal((:myerr, :libdoeye_caml), Cstring) |> unsafe_load
      throw(unsafe_string(cs))
      ccall((:flush_error, :libdoeye_caml), Cvoid, ())
    end
  end |> esc
end
