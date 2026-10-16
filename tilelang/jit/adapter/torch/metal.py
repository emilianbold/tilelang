from __future__ import annotations
from functools import wraps
from collections.abc import Callable

import torch
from tvm import tirx

from tilelang import tvm as tvm

from ..base import BaseKernelAdapter
from tilelang.engine.param import KernelParam


class MetalKernelAdapter(BaseKernelAdapter):
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        #  target: Union[str, Target],
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        #  host_mod: Optional[tvm.IRModule] = None,
        device_mod: tvm.IRModule | None = None,
        kernel_global_source: str | None = None,
        verbose: bool = False,
        #  pass_configs: Optional[Dict[str, Any]] = None,
        #  compile_flags: Optional[List[str]] = None
    ):
        self.kernel_global_source = kernel_global_source
        if isinstance(func_or_mod, tirx.PrimFunc):
            func_name = func_or_mod.attrs["global_symbol"]
        else:
            func_name = func_or_mod.__name__
        self.kernel_name = func_name + "_kernel"
        self.verbose = verbose

        self.block_info = [1, 1, 1]
        self.grid_info = [1, 1, 1]
        # SplitHostDevice collects the device function's parameters sorted by
        # name, while callers pass arguments in PrimFunc order. Record the
        # permutation so the launcher binds each argument to the buffer index
        # the kernel actually declares (None: orders already agree).
        self.arg_order: list[int] | None = None
        self._arg_order_error: str | None = None

        for var, func in device_mod.functions.items():
            assert var.name_hint == self.kernel_name
            thread_extent = func.attrs["thread_extent"]
            for tag, extent in thread_extent.items():
                if "threadIdx" in tag:
                    self.block_info["xyz".index(tag[-1])] = extent
                elif "blockIdx" in tag:
                    self.grid_info["xyz".index(tag[-1])] = extent
            if isinstance(func_or_mod, tirx.PrimFunc):
                try:
                    self.arg_order = self._compute_arg_order(func_or_mod, func)
                except ValueError as e:
                    # Defer: the kernel source is still usable without launching.
                    self._arg_order_error = str(e)
            break
        else:
            raise AssertionError(f"no kernel with name {func_name}")

        # print(self.block_info, self.grid_info)
        super().__init__(func_or_mod, result_idx=result_idx, params=params)

    _kernel = None

    @staticmethod
    def _compute_arg_order(prim_func: tirx.PrimFunc, device_func: tirx.PrimFunc) -> list[int] | None:
        host_names = [prim_func.buffer_map[p].name if p in prim_func.buffer_map else p.name for p in prim_func.params]
        device_names = [p.name for p in device_func.params]
        if host_names == device_names:
            return None
        if sorted(host_names) != sorted(device_names) or len(set(host_names)) != len(host_names):
            raise ValueError(
                "Cannot map PrimFunc parameters to the Metal kernel's parameters by name: "
                f"PrimFunc has {host_names}, kernel has {device_names}."
            )
        return [host_names.index(name) for name in device_names]

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if kernel_only:
            # Return just the kernel function body, stripping Metal
            # module-level boilerplate (includes, structs, etc.).
            idx = self.kernel_global_source.find("kernel void ")
            if idx >= 0:
                return self.kernel_global_source[idx:]
        return self.kernel_global_source

    def _convert_torch_func(self) -> Callable:
        if self._kernel is None:
            _kernel = getattr(torch.mps.compile_shader(self.kernel_global_source), self.kernel_name)
            _threads = [x * y for (x, y) in zip(self.block_info, self.grid_info)]
            _order = self.arg_order
            _order_error = self._arg_order_error

            @wraps(_kernel)
            def launcher(*args: torch.Tensor):
                if _order_error is not None:
                    raise ValueError(_order_error)
                if _order is not None:
                    args = tuple(args[i] for i in _order)
                return _kernel(
                    *args,
                    threads=_threads,
                    group_size=self.block_info,
                )

            self._kernel = launcher

        return self._kernel
