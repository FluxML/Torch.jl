(* Automatically generate the C++ -> C bindings.
   This takes as input the Descriptions.yaml file that gets generated when
   building PyTorch from source.
 *)
open Base
open Stdio

let excluded_functions =
  Set.of_list
    (module String)
    [ "multi_margin_loss"
    ; "multi_margin_loss_out"
    ; "log_softmax_backward_data"
    ; "softmax_backward_data"
    ; "copy_"
    ; "conv_transpose2d_backward_out"
    ; "conv_transpose3d_backward_out"
    ; "slow_conv_transpose2d_backward_out"
    ; "slow_conv_transpose3d_backward_out"
    ; "slow_conv3d_backward_out"
    ; "normal"
    ; "_cufft_set_plan_cache_max_size"
    ; "_cufft_clear_plan_cache"
    ; "backward"
    ; "_backward"
    ; "set_data"
    ; "_amp_non_finite_check_and_unscale_"
    ; "_cummin_helper"
    ; "_cummax_helper"
    ; "retain_grad"
    ; "_validate_sparse_coo_tensor_args"
    ; "_validate_sparse_csr_tensor_args"
    ; "count_nonzero"
    ; "_assert_async"
    ; "gradient"
    ; "linalg_vector_norm"
    ; "linalg_vector_norm_out"
    ; "linalg_matrix_norm"
    ; "linalg_matrix_norm_out"
    ; "histogram"
    ; "histogram_out"
    ]

let no_tensor_options =
  Set.of_list
    (module String)
    [ "zeros_like"
    ; "empty_like"
    ; "full_like"
    ; "ones_like"
    ; "rand_like"
    ; "randint_like"
    ; "randn_like"
    ]

let excluded_prefixes = [ "thnn_"; "th_"; "_foreach"; "_amp_foreach"; "linalg_norm" ]
let excluded_suffixes = [ "_forward"; "_forward_out" ]
let yaml_error yaml ~msg = failwith [%string "%{msg}, %{Yaml.to_string_exn yaml}"]

let extract_bool = function
  | `Bool b -> b
  | `String "true" -> true
  | `String "false" -> false
  | yaml -> yaml_error yaml ~msg:"expected bool"

let extract_list = function
  | `A l -> l
  | yaml -> yaml_error yaml ~msg:"expected list"

let extract_map = function
  | `O map -> Map.of_alist_exn (module String) map
  | yaml -> yaml_error yaml ~msg:"expected map"

let extract_string = function
  | `String s -> s
  (* The yaml spec for torch uses n which is converted to a bool. *)
  | `Bool b -> if b then "y" else "n"
  | `Float f -> Float.to_string f
  | yaml -> yaml_error yaml ~msg:"expected string"

module Func = struct
  type arg_type =
    | Bool
    | Int64
    | Double
    | Tensor
    | TensorOption
    | IntList
    | TensorOptList
    | TensorList
    | TensorOptions
    | Scalar
    | ScalarType
    | Device
    | String

  type arg =
    { arg_name : string
    ; arg_type : arg_type
    ; default_value : string option
    }

  type t =
    { name : string
    ; operator_name : string
    ; overload_name : string
    ; args : arg list
    ; returns : (* number of tensors that are returned *)
        [ `fixed of int | `dynamic ]
    ; kind : [ `function_ | `method_ ]
    }

  let arg_type_of_string str ~is_nullable =
    match String.lowercase str with
    | "bool" -> Some Bool
    | "int64_t" -> Some Int64
    | "double" -> Some Double
    | "at::tensor" -> Some (if is_nullable then TensorOption else Tensor)
    | "at::tensoroptions" -> Some TensorOptions
    | "at::intarrayref" | "intlist" -> Some IntList
    | "const c10::list<c10::optional<at::tensor>> &" -> Some TensorOptList
    | "at::tensorlist" -> Some TensorList
    | "at::device" -> Some Device
    | "at::scalar" | "const at::scalar &" -> Some Scalar
    | "at::scalartype" -> Some ScalarType
    | "c10::string_view" -> Some String
    | _ -> None

  let c_typed_args_list t =
    List.map t.args ~f:(fun { arg_name; arg_type; _ } ->
        match arg_type with
        | IntList -> [%string "int64_t *%{arg_name}_data, int %{arg_name}_len"]
        | TensorOptList | TensorList ->
          [%string "tensor *%{arg_name}_data, int %{arg_name}_len"]
        | TensorOptions -> [%string "int %{arg_name}_kind, int %{arg_name}_device"]
        | otherwise ->
          let simple_type_cstring =
            match otherwise with
            | Bool -> "int"
            | Int64 -> "int64_t"
            | Double -> "double"
            | Tensor -> "tensor"
            | TensorOption -> "tensor"
            | ScalarType -> "int"
            | Device -> "int"
            | Scalar -> "scalar"
            | String -> "char *"
            | IntList | TensorOptList | TensorList | TensorOptions -> assert false
          in
          Printf.sprintf "%s %s" simple_type_cstring arg_name)
    |> String.concat ~sep:", "

  let c_args_list args =
    List.map args ~f:(fun { arg_name; arg_type; _ } ->
        match arg_type with
        | Scalar | Tensor -> "*" ^ arg_name
        | TensorOption -> [%string "(%{arg_name} ? *%{arg_name} : torch::Tensor())"]
        | Bool -> "(bool)" ^ arg_name
        | IntList -> [%string "torch::IntArrayRef(%{arg_name}_data, %{arg_name}_len)"]
        | String -> [%string "std::string(%{arg_name})"]
        | TensorList -> [%string "of_carray_tensor(%{arg_name}_data, %{arg_name}_len)"]
        | TensorOptList ->
          Printf.sprintf "of_carray_tensor_opt(%s_data, %s_len)" arg_name arg_name
        | TensorOptions ->
          [%string
            "at::device(device_of_int(%{arg_name}_device)).dtype(at::ScalarType(%{arg_name}_kind))"]
        | ScalarType -> [%string "torch::ScalarType(%{arg_name})"]
        | Device -> [%string "device_of_int(%{arg_name})"]
        | _ -> arg_name)
    |> String.concat ~sep:", "

  let c_call t =
    match t.kind with
    | `function_ -> [%string "torch::%{t.name}(%{c_args_list t.args})"]
    | `method_ ->
      (match t.args with
      | head :: tail -> [%string "%{head.arg_name}->%{t.name}(%{c_args_list tail})"]
      | [] ->
        failwith [%string "Method calls should have at least one argument %{t.name}"])
end

exception Not_a_simple_arg

let read_yaml filename =
  let funcs =
    (* Split the file to avoid Yaml.of_string_exn segfaulting. *)
    In_channel.with_file filename ~f:In_channel.input_lines
    |> List.group ~break:(fun _ l -> String.length l > 0 && Char.( = ) l.[0] '-')
    |> List.concat_map ~f:(fun lines ->
           Yaml.of_string_exn (String.concat lines ~sep:"\n") |> extract_list)
  in
  printf "Read %s, got %d functions.\n%!" filename (List.length funcs);
  List.filter_map funcs ~f:(fun yaml ->
      let map = extract_map yaml in
      let name = Map.find_exn map "name" |> extract_string in
      let operator_name = Map.find_exn map "operator_name" |> extract_string in
      let overload_name = Map.find_exn map "overload_name" |> extract_string in
      let deprecated = Map.find_exn map "deprecated" |> extract_bool in
      let method_of =
        Map.find_exn map "method_of" |> extract_list |> List.map ~f:extract_string
      in
      let arguments = Map.find_exn map "arguments" |> extract_list in
      let returns =
        let is_tensor returns =
          let returns = extract_map returns in
          let return_type = Map.find_exn returns "dynamic_type" |> extract_string in
          String.( = ) return_type "at::Tensor"
        in
        let returns = Map.find_exn map "returns" |> extract_list in
        if List.for_all returns ~f:is_tensor
        then Some (`fixed (List.length returns))
        else (
          match returns with
          | [ returns ] ->
            let return_type =
              Map.find_exn (extract_map returns) "dynamic_type" |> extract_string
            in
            if String.( = ) return_type "at::TensorList"
               || String.( = )
                    return_type
                    "dynamic_type: const c10::List<c10::optional<Tensor>> &"
            then Some `dynamic
            else None
          | [] | _ :: _ :: _ -> None)
      in
      let kind =
        if List.exists method_of ~f:(String.( = ) "namespace")
        then Some `function_
        else if List.exists method_of ~f:(String.( = ) "Tensor")
        then Some `method_
        else None
      in
      if (not deprecated)
         && (not
               (List.exists excluded_prefixes ~f:(fun prefix ->
                    String.is_prefix name ~prefix)))
         && (not
               (List.exists excluded_suffixes ~f:(fun suffix ->
                    String.is_suffix name ~suffix)))
         && not (Set.mem excluded_functions name)
      then
        Option.both returns kind
        |> Option.bind ~f:(fun (returns, kind) ->
               try
                 let args =
                   List.filter_map arguments ~f:(fun arg ->
                       let arg = extract_map arg in
                       let arg_name = Map.find_exn arg "name" |> extract_string in
                       let arg_type = Map.find_exn arg "dynamic_type" |> extract_string in
                       let is_nullable =
                         Map.find arg "is_nullable"
                         |> Option.value_map ~default:false ~f:extract_bool
                       in
                       let default_value =
                         Map.find arg "default" |> Option.map ~f:extract_string
                       in
                       match Func.arg_type_of_string arg_type ~is_nullable with
                       | Some Scalar when Option.is_some default_value && not is_nullable
                         -> None
                       | Some TensorOptions
                         when Option.is_some default_value
                              && Set.mem no_tensor_options name -> None
                       | Some arg_type -> Some { Func.arg_name; arg_type; default_value }
                       | None ->
                         if Option.is_some default_value
                         then None
                         else raise Not_a_simple_arg)
                 in
                 Some { Func.name; operator_name; overload_name; args; returns; kind }
               with
               | Not_a_simple_arg -> None)
      else None)

let p out_channel s =
  Printf.ksprintf
    (fun line ->
      Out_channel.output_string out_channel line;
      Out_channel.output_char out_channel '\n')
    s

let write_cpp funcs filename =
  Out_channel.with_file (filename ^ ".cpp.h") ~f:(fun out_cpp ->
      Out_channel.with_file (filename ^ ".h") ~f:(fun out_h ->
          let pc s = p out_cpp s in
          let ph s = p out_h s in
          pc "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
          pc "";
          ph "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!";
          ph "";
          Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
              let c_typed_args_list = Func.c_typed_args_list func in
              match func.returns with
              | `dynamic ->
                pc "int atg_%s(tensor *out__, %s) {" exported_name c_typed_args_list;
                pc "  PROTECT(";
                pc "    auto outputs__ = %s;" (Func.c_call func);
                (* the returned type is a C++ vector of tensors *)
                pc "    int sz = outputs__.size();";
                pc "    for (int i = 0; i < sz; ++i)";
                pc "      out__[i] = new torch::Tensor(outputs__[i]);";
                pc "    out__[sz] = nullptr;";
                pc "    return 0;";
                pc "  )";
                pc "  return 1;";
                pc "}";
                pc "";
                ph "int atg_%s(tensor *, %s);" exported_name c_typed_args_list
              | `fixed ntensors ->
                pc "int atg_%s(tensor *out__, %s) {" exported_name c_typed_args_list;
                pc "  PROTECT(";
                pc "    auto outputs__ = %s;" (Func.c_call func);
                if ntensors = 1
                then pc "    out__[0] = new torch::Tensor(outputs__);"
                else
                  for i = 0 to ntensors - 1 do
                    pc "    out__[%d] = new torch::Tensor(std::get<%d>(outputs__));" i i
                  done;
                pc "    return 0;";
                pc "  )";
                pc "  return 1;";
                pc "}";
                pc "";
                ph "int atg_%s(tensor *, %s);" exported_name c_typed_args_list)))

let methods =
  let c name args =
    { Func.name
    ; operator_name = name
    ; overload_name = ""
    ; args
    ; returns = `fixed 1
    ; kind = `method_
    }
  in
  let ca arg_name arg_type = { Func.arg_name; arg_type; default_value = None } in
  [ c "grad" [ ca "self" Tensor ]
  ; c "set_requires_grad" [ ca "self" Tensor; ca "r" Bool ]
  ; c "toType" [ ca "self" Tensor; ca "scalar_type" ScalarType ]
  ; c "to" [ ca "self" Tensor; ca "device" Device ]
  ]

let run ~yaml_filename ~cpp_filename =
  let funcs = read_yaml yaml_filename in
  let funcs = methods @ funcs in
  printf "Generating code for %d functions.\n%!" (List.length funcs);
  (* Generate some unique names for overloaded functions. *)
  let funcs =
    List.map funcs ~f:(fun func -> String.lowercase func.operator_name, func)
    |> Map.of_alist_multi (module String)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (name, funcs) ->
           match funcs with
           | [] -> assert false
           | [ func ] -> [ name, func ]
           | funcs ->
             let has_empty_overload =
               List.exists funcs ~f:(fun (func : Func.t) ->
                   String.is_empty func.overload_name)
             in
             List.sort funcs ~compare:(fun (f1 : Func.t) (f2 : Func.t) ->
                 match Int.compare (String.length f1.name) (String.length f2.name) with
                 | 0 -> Int.compare (List.length f1.args) (List.length f2.args)
                 | cmp -> cmp)
             |> List.mapi ~f:(fun index (func : Func.t) ->
                    let operator_name = String.lowercase func.operator_name in
                    let overload_name = String.lowercase func.overload_name in
                    let name =
                      if String.is_empty overload_name
                         || (index = 0 && not has_empty_overload)
                      then operator_name
                      else if String.is_suffix operator_name ~suffix:"_"
                      then operator_name ^ overload_name ^ "_"
                      else operator_name ^ "_" ^ overload_name
                    in
                    name, func))
    |> Map.of_alist_exn (module String)
  in
  write_cpp funcs cpp_filename

let () =
  run
    ~yaml_filename:"data/Declarations.yaml"
    ~cpp_filename:"../c_wrapper/torch_api_generated"
