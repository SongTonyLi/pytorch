# mypy: allow-untyped-defs
"""
Global kernel cache for NVIDIA Universal GEMM.

This module provides a lazy-initialized cache for cutlass_api kernels,
avoiding expensive manifest scans on every kernel lookup.

The first call to get_kernel_by_name() loads all kernels from cutlass_api
(~10 seconds) and builds a name->kernel dict. Subsequent calls use the
dict for O(1) lookup (~0.1 μs).
"""

import logging
import threading
from collections.abc import Callable
from typing import Any


log = logging.getLogger(__name__)

_cache_lock = threading.Lock()

# Global cache: kernel_name -> kernel object
_kernel_by_name_cache: dict[str, Any] | None = None


def _build_kernel_cache() -> dict[str, Any]:
    """Build the kernel name -> kernel object cache."""
    import cutlass_api

    log.debug("Building NVGEMM kernel cache (this may take a few seconds)...")

    try:
        from torch._inductor.kernel.vendored_templates.cutedsl import (  # noqa: F401
            wrappers,
        )
    except ImportError:
        log.debug("Vendored kernel wrappers not available")

    all_kernels = cutlass_api.get_kernels()
    cache = {k.metadata.kernel_name: k for k in all_kernels}
    log.debug("NVGEMM kernel cache built: %d kernels", len(cache))
    return cache


def get_compatible_kernels(
    args: Any,
    cc: int,
    metadata_filter: Callable[[Any], bool] | None = None,
) -> list[Any]:
    """Get kernels compatible with the given arguments from the cache."""
    global _kernel_by_name_cache

    if _kernel_by_name_cache is None:
        with _cache_lock:
            if _kernel_by_name_cache is None:
                _kernel_by_name_cache = _build_kernel_cache()

    compatible = []
    for kernel in _kernel_by_name_cache.values():
        if kernel.metadata.min_cc > cc:
            continue

        if metadata_filter is not None and not metadata_filter(kernel.metadata):
            continue

        status = kernel.supports(args)
        if status.error is not None:
            continue
        compatible.append(kernel)

    log.debug(
        "Found %d compatible kernels from cache of %d total",
        len(compatible),
        len(_kernel_by_name_cache),
    )
    return compatible


def get_kernel_by_name(kernel_name: str) -> Any:
    """Get a cutlass_api kernel by name using the global cache."""
    global _kernel_by_name_cache

    if _kernel_by_name_cache is None:
        with _cache_lock:
            if _kernel_by_name_cache is None:
                _kernel_by_name_cache = _build_kernel_cache()

    return _kernel_by_name_cache.get(kernel_name)


def ensure_cache_initialized() -> None:
    """Ensure the kernel cache is initialized."""
    global _kernel_by_name_cache

    if _kernel_by_name_cache is None:
        with _cache_lock:
            if _kernel_by_name_cache is None:
                _kernel_by_name_cache = _build_kernel_cache()


# Cache for EFC kernels with specific epilogue configurations
# Key: (efc_kernel_name, epilogue_fn_code) -> kernel object
_efc_epilogue_cache: dict[tuple[str, str], Any] = {}


def clear_cache() -> None:
    """Clear all kernel caches."""
    global _kernel_by_name_cache, _efc_epilogue_cache
    with _cache_lock:
        _kernel_by_name_cache = None
        _efc_epilogue_cache = {}


class _NVGEMMCacheWrapper:
    """Wrapper to integrate with torch._inductor.utils.clear_caches()."""

    def cache_clear(self) -> None:
        clear_cache()


from torch._inductor.utils import clear_on_fresh_cache


clear_on_fresh_cache(_NVGEMMCacheWrapper())


def get_efc_kernel_with_epilogue(
    efc_kernel_name: str,
    epilogue_args: Any,
    epilogue_source: str = "",
) -> Any:
    """Get an EFC kernel configured with a specific epilogue.

    This avoids the slow get_kernels() call by:
    1. Getting the base kernel's metadata from the cache
    2. Creating a new kernel with epilogue metadata added

    Args:
        efc_kernel_name: The EFC kernel name (e.g., cutedsl.PersistentDenseGemmEFCKernel_...)
        epilogue_args: EpilogueArguments from cutlass_api
        epilogue_source: Source code string of the epilogue function (preferred cache key).
            When provided, avoids unreliable inspect.getsource on generated code.

    Returns:
        The configured EFC kernel, or None if not found.
    """
    global _kernel_by_name_cache

    if not epilogue_source:
        epilogue_source = str(epilogue_args) if epilogue_args is not None else ""

    cache_key = (efc_kernel_name, epilogue_source)

    # Hold lock for the full operation — this is not a hot path (called once per
    # unique kernel+epilogue combination), so simplicity beats concurrency here.
    with _cache_lock:
        if cache_key in _efc_epilogue_cache:
            log.debug("EFC kernel with epilogue found in cache: %s", efc_kernel_name)
            return _efc_epilogue_cache[cache_key]

        if _kernel_by_name_cache is None:
            _kernel_by_name_cache = _build_kernel_cache()

        base_kernel = _kernel_by_name_cache.get(efc_kernel_name)
        if base_kernel is None:
            log.debug("Base EFC kernel not found: %s", efc_kernel_name)
            return None

        from cutlass_api.metadata import EpilogueMetadata, KernelMetadata

        epilogue_metadata = EpilogueMetadata.from_args(epilogue_args)

        base_metadata = base_kernel.metadata
        new_metadata = KernelMetadata(
            operands=base_metadata.operands,
            design=base_metadata.design,
            kernel_name=base_metadata.kernel_name,
            kernel_class=base_metadata.kernel_class,
            min_cc=base_metadata.min_cc,
            epilogue=epilogue_metadata,
        )

        kernel_class = base_metadata.kernel_class
        new_kernel = kernel_class(new_metadata)

        _efc_epilogue_cache[cache_key] = new_kernel
        log.debug("Created and cached EFC kernel with epilogue: %s", efc_kernel_name)

        return new_kernel
