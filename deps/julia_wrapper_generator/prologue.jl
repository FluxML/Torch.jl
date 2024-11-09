function get_error()
  err = cglobal((:myerr, libtorch_c_api), Cstring) |> unsafe_load
  unsafe_string(err)
end

macro runtime_error_check(ex)
  quote
    x = $ex
    if x == 1
      cs = get_error()
      flush_error()
      throw(cs)
    end
  end |> esc
end
