macro runtime_error_check(ex)
  quote
    x = $ex
    if x == 1
      cs = cglobal((:myerr, :libdoeye_caml), Cstring) |> unsafe_load
      ccall((:flush_error, :libdoeye_caml), Cvoid, ())
      throw(unsafe_string(cs))
    end
  end |> esc
end
