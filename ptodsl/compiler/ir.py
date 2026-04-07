import inspect

from mlir.dialects import func, pto as _pto
from mlir.ir import Attribute, Context, InsertionPoint, Location, Module, UnitAttr

from ..api.scalar import wrap_value


_MODULE_STACK = []


class FuncRef:
    def __init__(self, sym_name):
        self.sym_name = sym_name


class _ModuleState:
    def __init__(self, *, ctx, module, meta_map):
        self.ctx = ctx
        self.module = module
        self.meta_map = meta_map


def _resolve_meta(meta_fn):
    values = meta_fn()
    if not isinstance(values, dict):
        raise ValueError(
            "`meta_data()` must return a dict of named symbols to MLIR/PTO types."
        )
    return dict(values)


def _resolve_arg_types(signature, meta_map):
    arg_types = []
    for param in signature.parameters.values():
        annot = param.annotation
        if isinstance(annot, str):
            if annot not in meta_map:
                raise ValueError(f"Unknown annotation '{annot}'.")
            arg_types.append(meta_map[annot])
        elif annot is inspect._empty:
            raise ValueError(f"Missing annotation for argument '{param.name}'.")
        else:
            arg_types.append(annot)
    return arg_types


def _resolve_ret_types(signature, meta_map):
    ret_annot = signature.return_annotation
    if ret_annot in (inspect._empty, None):
        return []
    if isinstance(ret_annot, str):
        if ret_annot not in meta_map:
            raise ValueError(f"Unknown return annotation '{ret_annot}'.")
        return [meta_map[ret_annot]]
    if isinstance(ret_annot, (list, tuple)):
        out = []
        for elem in ret_annot:
            if isinstance(elem, str):
                out.append(meta_map[elem])
            else:
                out.append(elem)
        return out
    return [ret_annot]


def _has_func_return(block):
    last_name = None
    for op in block.operations:
        last_name = op.operation.name
    return last_name == "func.return"


def _inject_globals(fn, values):
    old = {}
    for name, value in values.items():
        old[name] = fn.__globals__.get(name, None)
        fn.__globals__[name] = value
    return old


def _restore_globals(fn, old, injected_names):
    for name in injected_names:
        if old[name] is None and name in fn.__globals__:
            del fn.__globals__[name]
        else:
            fn.__globals__[name] = old[name]


def _build_func_body(ir_func, fn, ret_types, meta_map):
    entry = ir_func.add_entry_block()
    with InsertionPoint(entry):
        wrapped_args = [wrap_value(arg) for arg in entry.arguments]
        injected = set(meta_map.keys())
        old_globals = _inject_globals(fn, meta_map)
        try:
            fn(*wrapped_args)
        finally:
            _restore_globals(fn, old_globals, injected)

        if not ret_types and not _has_func_return(entry):
            func.ReturnOp([])


def _current_module_state():
    if not _MODULE_STACK:
        raise RuntimeError(
            "`pto.func(...)` can only be used inside `@to_ir_module(..., module=True)`."
        )
    return _MODULE_STACK[-1]


def ir_func(*, name=None, entry=False, kernel=None):
    def decorator(fn):
        state = _current_module_state()
        sig = inspect.signature(fn)
        arg_types = _resolve_arg_types(sig, state.meta_map)
        ret_types = _resolve_ret_types(sig, state.meta_map)
        fn_name = name or fn.__name__
        fn_ty = func.FunctionType.get(arg_types, ret_types)

        with InsertionPoint(state.module.body):
            ir_op = func.FuncOp(fn_name, fn_ty)

        if entry:
            ir_op.operation.attributes["pto.entry"] = UnitAttr.get(state.ctx)
        if kernel is not None:
            ir_op.operation.attributes["pto.kernel_kind"] = Attribute.parse(
                f"#pto.kernel_kind<{kernel}>"
            )

        _build_func_body(ir_op, fn, ret_types, state.meta_map)
        return FuncRef(fn_name)

    return decorator


def _build_single_func_module(fn, meta_map):
    sig = inspect.signature(fn)
    arg_types = _resolve_arg_types(sig, meta_map)
    ret_types = _resolve_ret_types(sig, meta_map)
    module = Module.create()
    fn_ty = func.FunctionType.get(arg_types, ret_types)

    with InsertionPoint(module.body):
        ir_op = func.FuncOp(fn.__name__, fn_ty)

    _build_func_body(ir_op, fn, ret_types, meta_map)
    return module


def _build_multi_func_module(fn, meta_map, ctx):
    if inspect.signature(fn).parameters:
        raise ValueError("`module=True` expects a zero-argument builder function.")

    module = Module.create()
    injected = set(meta_map.keys())
    old_globals = _inject_globals(fn, meta_map)
    _MODULE_STACK.append(_ModuleState(ctx=ctx, module=module, meta_map=meta_map))
    try:
        fn()
    finally:
        _MODULE_STACK.pop()
        _restore_globals(fn, old_globals, injected)
    return module


def to_ir_module(*, meta_data, module=False):
    def decorator(fn):
        with Context() as ctx, Location.unknown():
            _pto.register_dialect(ctx, load=True)
            meta_map = _resolve_meta(meta_data)
            if module:
                ir_module = _build_multi_func_module(fn, meta_map, ctx)
            else:
                ir_module = _build_single_func_module(fn, meta_map)
            ir_module.operation.verify()
            return ir_module

    return decorator


__all__ = ["FuncRef", "ir_func", "to_ir_module"]
