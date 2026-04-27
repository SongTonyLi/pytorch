# Owner(s): ["module: inductor"]
import os
import time
import unittest
from collections.abc import Callable
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
ORIGAMI_TOPK_VALUES = [5, 10]
ORIGAMI_COMPILE_TOPK = 2
PERF_SLOWDOWN_TOLERANCE = 1.05
IS_ROCM = torch.version.hip is not None

try:
    import origami

    HAS_ORIGAMI = True
except ImportError:
    origami = None
    HAS_ORIGAMI = False


if IS_ROCM:
    torch.set_float32_matmul_precision("highest")


@unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU and Triton")
@unittest.skipIf(not IS_ROCM, "Origami integration is ROCm-only")
@unittest.skipIf(not HAS_ORIGAMI, "Origami package is not installed")
class TestOrigami(TestCase):
    def _make_fn_and_inputs(
        self, op_name: str, size: int
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)

        if op_name == "bmm":
            batch = 4
            a = torch.randn(batch, size, size, device=GPU_TYPE, dtype=torch.float16)
            b = torch.randn(batch, size, size, device=GPU_TYPE, dtype=torch.float16)

            def fn(x, y):
                return torch.bmm(x, y)

            return fn, (a, b)

        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)

        if op_name == "mm":

            def fn(x, y):
                return torch.mm(x, y)

            return fn, (a, b)

        if op_name == "addmm":
            bias = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)

            def fn(inp, x, y):
                return torch.addmm(inp, x, y)

            return fn, (bias, a, b)

        raise AssertionError(f"Unsupported op {op_name}")

    def _benchmark_gpu_call_count(self) -> int:
        return sum(
            value
            for name, value in counters["inductor"].items()
            if "benchmark_gpu" in name
        )

    def _measure_compile_time(
        self, op_name: str, patch_config: dict[str, object], *, size: int
    ) -> float:
        """Return first-call torch.compile wall time (seconds).

        fresh_cache() redirects both the inductor and Triton cache to a fresh
        temp directory, giving clean isolated compile-time numbers.
        """
        fn, args = self._make_fn_and_inputs(op_name, size)
        torch._dynamo.reset()
        with fresh_cache(), config.patch(patch_config):
            compiled = torch.compile(fn, dynamic=False)
            t0 = time.perf_counter()
            _ = compiled(*args)
            torch.cuda.synchronize()
            return time.perf_counter() - t0

    def _compile_with_config(
        self,
        op_name: str,
        patch_config: dict[str, object],
        *,
        size: int,
    ) -> dict[str, object]:
        fn, args = self._make_fn_and_inputs(op_name, size)
        expected = fn(*args)

        torch._dynamo.reset()
        counters.clear()

        with (
            fresh_cache(),
            config.patch(patch_config),
            mock.patch(
                "origami.select_topk_configs", wraps=origami.select_topk_configs
            ) as select_topk,
        ):
            compiled = torch.compile(fn, dynamic=False)
            result = compiled(*args)

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

        return {
            "compiled": compiled,
            "args": args,
            "benchmark_gpu_calls": self._benchmark_gpu_call_count(),
            "topk_calls": select_topk.call_count,
        }

    def _origami_default_config(self, topk: int) -> dict[str, object]:
        return {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "rocm.origami": True,
            "rocm.origami_topk": topk,
            "max_autotune_gemm_search_space": "DEFAULT",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    def _origami_exhaustive_config(self) -> dict[str, object]:
        return {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "rocm.origami": True,
            "rocm.origami_topk": ORIGAMI_TOPK_VALUES[0],
            "max_autotune_gemm_search_space": "EXHAUSTIVE",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    def _max_autotune_default_config(self) -> dict[str, object]:
        return {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "rocm.origami": False,
            "max_autotune_gemm_search_space": "DEFAULT",
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    def test_origami_respects_gemm_search_space(self):
        for op_name in ("mm", "addmm", "bmm"):
            with self.subTest(op_name=op_name, search_space="DEFAULT"):
                default_case = self._compile_with_config(
                    op_name,
                    self._origami_default_config(ORIGAMI_TOPK_VALUES[0]),
                    size=256,
                )
                self.assertGreater(default_case["topk_calls"], 0)

            with self.subTest(op_name=op_name, search_space="EXHAUSTIVE"):
                exhaustive_case = self._compile_with_config(
                    op_name,
                    self._origami_exhaustive_config(),
                    size=256,
                )
                self.assertEqual(exhaustive_case["topk_calls"], 0)

    def test_origami_reduces_compile_work_vs_regular_max_autotune(self):
        for op_name in ("mm", "addmm", "bmm"):
            with self.subTest(op_name=op_name):
                origami_time = self._measure_compile_time(
                    op_name,
                    {
                        **self._origami_default_config(ORIGAMI_COMPILE_TOPK),
                    },
                    size=256,
                )
                max_autotune_time = self._measure_compile_time(
                    op_name,
                    self._max_autotune_default_config(),
                    size=256,
                )
                self.assertLess(origami_time, max_autotune_time)

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_origami_runtime_matches_regular_max_autotune(self):
        for op_name in ("mm", "addmm", "bmm"):
            for size in (8192, 16384):
                for topk in ORIGAMI_TOPK_VALUES:
                    with self.subTest(op_name=op_name, size=size, topk=topk):
                        origami_case = self._compile_with_config(
                            op_name,
                            self._origami_default_config(topk),
                            size=size,
                        )
                        max_autotune_case = self._compile_with_config(
                            op_name,
                            self._max_autotune_default_config(),
                            size=size,
                        )

                        origami_runtime_ms = benchmarker.benchmark(
                            origami_case["compiled"],
                            origami_case["args"],
                            {},
                            warmup=50,
                            rep=200,
                        )
                        max_autotune_runtime_ms = benchmarker.benchmark(
                            max_autotune_case["compiled"],
                            max_autotune_case["args"],
                            {},
                            warmup=50,
                            rep=200,
                        )

                        print(
                            f"{op_name} size={size} topk={topk} runtime ms: "
                            f"origami={origami_runtime_ms:.3f}, "
                            f"max_autotune={max_autotune_runtime_ms:.3f}"
                        )

                        self.assertLessEqual(
                            origami_runtime_ms,
                            max_autotune_runtime_ms * PERF_SLOWDOWN_TOLERANCE,
                        )


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON and IS_ROCM:
        run_tests()
