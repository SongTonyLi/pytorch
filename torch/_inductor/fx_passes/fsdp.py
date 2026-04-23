import logging
from collections.abc import Callable

import torch
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    bucket_reduce_scatter_by_mb,
    BucketMode,
    is_all_gather_into_tensor as is_all_gather,
    merge_all_gather,
    merge_reduce_scatter,
)
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def is_fsdp_all_gather(n):
    assert is_all_gather(n)
    while len(n.all_input_nodes) == 1:
        n = n.all_input_nodes[0]
        if n.op == "placeholder":
            return True
    return False


def is_fsdp_all_gather_wait(wait: torch.fx.Node) -> bool:
    # Assume all_gather_into_tensor input is either graph input
    # or dtype conversion of graph input
    ag_node = wait.args[0]  # type: ignore[arg-type, union-attr]
    return is_fsdp_all_gather(ag_node)


def is_graph_output(node: torch.fx.Node) -> bool:
    return all(user.op == "output" for user in node.users)


def is_fsdp_reduce_scatter_wait(wait: torch.fx.Node) -> bool:
    if is_graph_output(wait):
        return True

    if len(wait.users) == 1:
        user = next(iter(wait.users))
        assert user is not None
        return (
            is_graph_output(user)
            and user.op == "call_function"
            and user.target is torch.ops.prims.convert_element_type.default
        )

    return False


_c10d = torch.ops._c10d_functional
_aten = torch.ops.aten
_LINEAR_REDUCE_OPS = OrderedSet(["sum", "avg"])

_dedup_rs_pass = PatternMatcherPass(pass_name="dedup_reduce_scatter")


def _wait_rs(name: str) -> CallFunction:
    """Build a pattern matching wait_tensor(reduce_scatter_tensor(name, ...))."""
    return CallFunction(
        _c10d.wait_tensor.default,
        CallFunction(
            _c10d.reduce_scatter_tensor.default,
            KeywordArg(name),
            KeywordArg("reduce_op"),
            KeywordArg("group_size"),
            KeywordArg("group_name"),
        ),
    )


def _dedup_rs_extra_check(match: Match) -> bool:
    """Reject matches with non-linear reduce ops, multi-user RS/wait nodes, or mismatched dtypes."""
    if match.kwargs["reduce_op"] not in _LINEAR_REDUCE_OPS:
        return False
    for node in match.nodes:
        if node.target is _aten.add.Tensor:
            continue
        if node.target not in (
            _c10d.wait_tensor.default,
            _c10d.reduce_scatter_tensor.default,
        ):
            return False
        if len(node.users) != 1:
            return False
    input_a = match.kwargs["input_a"]
    input_b = match.kwargs["input_b"]
    if input_a.meta["val"].dtype != input_b.meta["val"].dtype:
        return False
    return True


@register_graph_pattern(
    CallFunction(
        _aten.add.Tensor,
        _wait_rs("input_a"),
        _wait_rs("input_b"),
    ),
    extra_check=_dedup_rs_extra_check,
    # pyrefly: ignore[bad-argument-type]
    pass_dict=_dedup_rs_pass,
)
def _dedup_rs_handler(
    match: Match, input_a, input_b, reduce_op, group_size, group_name
):
    """Replace add(wait(rs(a)), wait(rs(b))) with wait(rs(add(a, b)))."""

    def repl(input_a, input_b):
        combined = _aten.add.Tensor(input_a, input_b)
        rs = _c10d.reduce_scatter_tensor.default(
            combined, reduce_op, group_size, group_name
        )
        return _c10d.wait_tensor.default(rs)

    # pyrefly: ignore[bad-argument-type]
    match.replace_by_example(repl, [input_a, input_b])


def dedup_fsdp_reduce_scatter(gm: torch.fx.GraphModule) -> None:
    """
    Fuse duplicate reduce_scatter ops whose waited results are summed.

    RS is linear, so RS(a) + RS(b) = RS(a + b). This pass rewrites
        rs_a = reduce_scatter(input_a, ...); wait_a = wait(rs_a)
        rs_b = reduce_scatter(input_b, ...); wait_b = wait(rs_b)
        result = add(wait_a, wait_b)
    into
        combined = add(input_a, input_b)
        rs = reduce_scatter(combined, ...)
        result = wait(rs)

    For N-way add trees (N > 2), the pattern is applied repeatedly
    until fixpoint — each iteration fuses one leaf pair.
    """
    while _dedup_rs_pass.apply(gm):
        pass
    gm.graph.lint()
    gm.recompile()


def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket id and returns size of a bucket in megabytes.
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    assert bucket_cap_mb_by_bucket_idx is not None
    ag_buckets = bucket_all_gather_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_all_gather_wait,
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_fsdp_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP reduce_scatter ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket idx and returns size of a bucket in megabytes. By default
            torch._inductor.fx_passes.bucketing.bucket_cap_mb_by_bucket_idx_default is used.

    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    rs_buckets = bucket_reduce_scatter_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_reduce_scatter_wait,
    )
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)
