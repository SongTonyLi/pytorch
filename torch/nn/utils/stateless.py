# mypy: allow-untyped-defs
import contextlib
from collections import defaultdict
from typing import Any
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


__all__ = ["functional_call"]


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(inner) for inner in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(inner) for inner in value)
    return False


def _untie_named_tensors_map(
    module: "torch.nn.Module",
    parameters_and_buffers: dict[str, Tensor],
) -> dict[str, Tensor]:
    """
    Unties all tied tensors in the module to parameters_and_buffers.

    This function returns a new untied_parameters_and_buffers dictionary and leave the original
    untied_parameters_and_buffers dictionary unchanged. It adds new (missing) keys for tied tensors
    in the module to untied_parameters_and_buffers. The value of the new key is the user-given value
    in the original parameters_and_buffers dictionary.

    If there are more than one user-given values for the same tied tensor, it will raise an error.

    For example, if the module has two tied weights self.foo and self.tied_foo and the user passes
    {'foo': foo_value, ...}, this will return {'foo': foo_value, 'tied_foo': foo_value, ...}. If the
    user passes {'foo': foo_value, 'tied_foo': tied_foo_value, ...}, it will raise an error. If the
    user passes {'foo': foo_value, 'tied_foo': foo_value, ...}, it will not raise an error.

    Args:
        module (torch.nn.Module): the module to determine which tensors are tied.
        parameters_and_buffers (Dict[str, Tensor]): a map of {name: tensor} for reparamaterizing the module.

    Returns:
        A new untied version of the parameters_and_buffers dictionary.

    Raises:
        ValueError: if there are more than one user-given values for the same tied tensor.
    """
    # A map of {name: tensor} for all tensors (including tied ones) in the module.
    all_named_tensors: dict[str, Tensor] = {}
    all_named_tensors.update(module.named_parameters(remove_duplicate=False))
    all_named_tensors.update(module.named_buffers(remove_duplicate=False))

    # A map of {tensor: set(all_tied_names)} for all tensor names in the module.
    tensor_to_tied_names_map: dict[Tensor, set[str]] = {}
    for name, tensor in all_named_tensors.items():
        if tensor not in tensor_to_tied_names_map:
            tensor_to_tied_names_map[tensor] = set()
        tensor_to_tied_names_map[tensor].add(name)

    # A map of {tied_name: set(all_tied_names)} for all tensor names in the module.
    # If a name is not tied, it will not be in this map.
    tied_names_map: dict[str, set[str]] = {}
    for tied_names in tensor_to_tied_names_map.values():
        if len(tied_names) > 1:
            for tied_name in tied_names:
                tied_names_map[tied_name] = tied_names

    # Make sure the user didn't pass multiple values for the same tied tensor.
    given_names = set(parameters_and_buffers.keys())
    # same as given_names.intersection(tied_names_map.keys()) but dynamo can't
    # handle that
    given_names_for_tied_tensors: set[str] = set()
    for name in given_names:
        if name in tied_names_map:
            given_names_for_tied_tensors.add(name)

    for given_name in given_names_for_tied_tensors:
        tied_names = tied_names_map[given_name]
        if (
            # Detect if there are multiple keys present for the same tied tensor.
            len(tied_names.intersection(given_names_for_tied_tensors)) > 1
            # Only raise an error if the user passed multiple values for the same tied tensor.
            # If all given values are the same, don't raise.
            and len({parameters_and_buffers[tied_name] for tied_name in tied_names})
            != 1
        ):
            raise ValueError(
                f"functional_call got multiple values for keys {sorted(tied_names)}, "
                f"which are tied. Consider using tie_weights=False"
            )

    # Untie the given named tensor map
    # Make a copy for not modifying the original dict
    untied_parameters_and_buffers = parameters_and_buffers.copy()
    for given_name in given_names_for_tied_tensors:
        for tied_name in tied_names_map[given_name]:
            untied_parameters_and_buffers[tied_name] = parameters_and_buffers[
                given_name
            ]
    return untied_parameters_and_buffers


@contextlib.contextmanager
def _reparametrize_module(
    module: "torch.nn.Module",
    parameters_and_buffers: dict[str, Tensor],
    tie_weights: bool = False,
    strict: bool = False,
    stack_weights: bool = False,
):
    if tie_weights:
        untied_parameters_and_buffers = _untie_named_tensors_map(
            module, parameters_and_buffers
        )
    else:
        untied_parameters_and_buffers = parameters_and_buffers

    accessor = NamedMemberAccessor(module)
    if strict:
        missing_keys, unexpected_keys = accessor.check_keys(
            untied_parameters_and_buffers
        )
        error_msgs = []
        if len(unexpected_keys) > 0:
            error_msgs.append(
                f"Unexpected key(s): {', '.join(map(repr, unexpected_keys))}."
            )
        if len(missing_keys) > 0:
            error_msgs.append(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in reparametrizing for {}:\n\t{}".format(
                    module._get_name(), "\n\t".join(error_msgs)
                )
            )

    orig_parameters_and_buffers: dict[str, Tensor] = {}
    try:
        orig_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            untied_parameters_and_buffers, allow_missing=True
        )
        yield
    finally:
        if stack_weights:
            # When stacking is enabled, we will restore the weights in LIFO order.
            orig_parameters_and_buffers = dict(
                reversed(orig_parameters_and_buffers.items())
            )
        new_parameters_and_buffers, _ = accessor.swap_tensors_dict(
            orig_parameters_and_buffers, allow_missing=True
        )
        # Sometimes the module is not completely stateless and has some in-place modifications on
        # the _parameters and _buffers dictionaries.
        # Write the changed parameters and buffers back to the original dict.
        parameters_and_buffers.update(
            {
                k: new_parameters_and_buffers[k]
                for k in parameters_and_buffers
                if k in new_parameters_and_buffers
            }
        )


def _prepare_optimizer_reparametrization(
    optimizer: "torch.optim.Optimizer",
    parameters_and_buffers: dict[str, Tensor],
    optimizer_state_dict: dict[str, Any],
):
    """
    Validate and normalize optimizer state for ``_reparametrize_optimizer``.

    This follows the same structural assumptions as DCP-compatible optimizers,
    but consumes the raw ``optimizer.state_dict()`` format:
    ``state`` is keyed by packed parameter ids and each param group contains
    the live optimizer group fields plus a packed ``params`` list whose order
    matches ``optimizer.param_groups``.
    """
    if not optimizer.state:
        raise RuntimeError(
            "_reparametrize_optimizer requires initialized optimizer state."
        )
    if not isinstance(optimizer_state_dict, dict):
        raise RuntimeError(
            "_reparametrize_optimizer requires a DCP-style optimizer state_dict."
        )

    state = optimizer_state_dict.get("state")
    param_groups = optimizer_state_dict.get("param_groups")
    if not isinstance(state, dict) or not isinstance(param_groups, list):
        raise RuntimeError(
            "_reparametrize_optimizer requires an optimizer.state_dict()-style "
            "state_dict with 'state' and 'param_groups' entries."
        )
    if any(isinstance(name, torch.Tensor) for name in state):
        raise RuntimeError(
            "_reparametrize_optimizer requires optimizer.state_dict()-style "
            "state keyed by packed parameter ids."
        )
    if len(optimizer.param_groups) != len(param_groups):
        raise RuntimeError(
            "optimizer_state_dict has a different number of parameter groups than "
            "the live optimizer."
        )

    group_rebind_infos = []
    # Raw optimizer state_dicts address parameters by packed integer ids, so we
    # align explicit parameter tensors with optimizer.param_groups by order.
    # Example: if param_groups[*]["params"] is [[0, 1], [2]] and
    # parameters_and_buffers.values() is [fake_p0, fake_p1, fake_p2], then the
    # first optimizer group is rebound to [fake_p0, fake_p1] and the second to
    # [fake_p2].
    flat_parameters = list(parameters_and_buffers.values())
    flat_param_offset = 0
    packed_param_ids: set[int] = set()
    for idx, (group, saved_group) in enumerate(
        zip(optimizer.param_groups, param_groups, strict=True)
    ):
        if not isinstance(saved_group, dict):
            raise RuntimeError(
                "_reparametrize_optimizer requires each optimizer param group "
                "to be a dictionary."
            )
        names = saved_group.get("params")
        if not isinstance(names, list) or not all(
            isinstance(param_id, int) for param_id in names
        ):
            raise RuntimeError(
                "_reparametrize_optimizer requires optimizer.state_dict()-style "
                "param_groups[*]['params'] entries keyed by packed parameter ids."
            )
        if len(group["params"]) != len(names):
            raise RuntimeError(
                "optimizer_state_dict param group does not match the size of "
                f"live optimizer param group {idx}."
            )
        next_offset = flat_param_offset + len(names)
        if next_offset > len(flat_parameters):
            raise RuntimeError(
                "_reparametrize_optimizer requires the explicit parameter state to "
                "match optimizer.param_groups ordering."
            )
        # Slice out the explicit tensors that should back this optimizer group.
        rebind_params = flat_parameters[flat_param_offset:next_offset]
        flat_param_offset = next_offset

        for param_id in names:
            packed_param_ids.add(param_id)
            param_state = state.get(param_id, {})
            if not isinstance(param_state, dict):
                raise RuntimeError(
                    "_reparametrize_optimizer requires per-parameter optimizer "
                    "state entries to be dictionaries."
                )

        missing_group_keys = [
            key for key in saved_group if key != "params" and key not in group
        ]
        if missing_group_keys:
            raise RuntimeError(
                "_reparametrize_optimizer requires optimizer.state_dict()-style "
                "param group keys to match the live optimizer group keys. "
                f"Missing live keys for group {idx}: {missing_group_keys}"
            )

        group_rebind_infos.append(
            (
                group,  # live optimizer group to mutate
                saved_group,  # serialized group values to install temporarily
                rebind_params,  # explicit tensors that replace group["params"]
                {
                    key: group[key] for key in saved_group if key != "params"
                },  # restore data
            )
        )

    if flat_param_offset != len(flat_parameters):
        raise RuntimeError(
            "_reparametrize_optimizer requires the explicit parameter state to "
            "match optimizer.param_groups ordering."
        )

    if any(key not in packed_param_ids for key in state):
        raise RuntimeError(
            "_reparametrize_optimizer requires optimizer.state_dict()-style state "
            "to contain only per-parameter entries keyed by packed parameter ids."
        )
    return state, group_rebind_infos


@contextlib.contextmanager
def _reparametrize_optimizer(
    optimizer: "torch.optim.Optimizer",
    parameters_and_buffers: dict[str, Tensor],
    optimizer_state_dict: dict[str, Any],
):
    """
    Temporarily rebind an optimizer to explicit parameter tensors.

    ``optimizer_state_dict`` must be in the raw ``optimizer.state_dict()``
    format. This helper assumes a DCP-compatible optimizer structure, but it
    consumes the optimizer-native packed-param-id representation rather than
    DCP's FQN-keyed exported state.
    Tensor-valued optimizer state is rebound directly without cloning, so
    in-place optimizer updates may be reflected into ``optimizer_state_dict``.
    This is acceptable because this helper is intended for tracing-style
    explicit-state paths, where the provided optimizer state is treated as a
    mutable tracing input rather than an immutable snapshot.
    """
    state, group_rebind_infos = _prepare_optimizer_reparametrization(
        optimizer, parameters_and_buffers, optimizer_state_dict
    )

    original_state = optimizer.state
    original_group_params = [group["params"] for group in optimizer.param_groups]

    try:
        rebind_state: defaultdict[Tensor, Any] = defaultdict(dict)

        for group, saved_group, rebind_params, _ in group_rebind_infos:
            # Rebind the live optimizer group to the explicit tensors and saved
            # group metadata for the trace region.
            group["params"] = rebind_params
            for key, value in saved_group.items():
                if key == "params":
                    continue
                group[key] = value

            for rebind_param, param_id in zip(
                group["params"], saved_group["params"], strict=True
            ):
                # Re-key per-parameter optimizer state from packed ids to the
                # rebound parameter tensors expected by the live optimizer.
                # Example: if saved_group["params"] is [0, 1] and group["params"]
                # is [fake_p0, fake_p1], then state[0] becomes optimizer.state[fake_p0]
                # and state[1] becomes optimizer.state[fake_p1].
                rebind_state[rebind_param] = state.get(param_id, {})

        optimizer.state = rebind_state
        yield
    finally:
        # Restore the original live optimizer object exactly.
        for group, params in zip(
            optimizer.param_groups, original_group_params, strict=True
        ):
            group["params"] = params
        for group, _, _, saved_values in group_rebind_infos:
            for key, value in saved_values.items():
                group[key] = value
        optimizer.state = original_state


@deprecated(
    "`torch.nn.utils.stateless.functional_call` is deprecated as of PyTorch 2.0 "
    "and will be removed in a future version of PyTorch. "
    "Please use `torch.func.functional_call` instead which is a drop-in replacement.",
    category=FutureWarning,
)
def functional_call(
    module: "torch.nn.Module",
    parameters_and_buffers: dict[str, Tensor],
    args: Any | tuple | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    r"""Perform a functional call on the module by replacing the module parameters and buffers with the provided ones.

    .. warning::

        This API is deprecated as of PyTorch 2.0 and will be removed in a future
        version of PyTorch. Please use :func:`torch.func.functional_call` instead,
        which is a drop-in replacement for this API.

    .. note:: If the module has active parametrizations, passing a value in the
        :attr:`parameters_and_buffers` argument with the name set to the regular parameter
        name will completely disable the parametrization.
        If you want to apply the parametrization function to the value passed
        please set the key as ``{submodule_name}.parametrizations.{parameter_name}.original``.

    .. note:: If the module performs in-place operations on parameters/buffers, these will be reflected
        in the `parameters_and_buffers` input.

        Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # does self.foo = self.foo + 1
            >>> print(mod.foo)  # tensor(0.)
            >>> functional_call(mod, a, torch.ones(()))
            >>> print(mod.foo)  # tensor(0.)
            >>> print(a['foo'])  # tensor(1.)

    .. note:: If the module has tied weights, whether or not functional_call respects the tying is determined by the
        tie_weights flag.

        Example::

            >>> a = {'foo': torch.zeros(())}
            >>> # xdoctest: +SKIP
            >>> mod = Foo()  # has both self.foo and self.foo_tied which are tied. Returns x + self.foo + self.foo_tied
            >>> print(mod.foo)  # tensor(1.)
            >>> mod(torch.zeros(()))  # tensor(2.)
            >>> functional_call(mod, a, torch.zeros(()))  # tensor(0.) since it will change self.foo_tied too
            >>> functional_call(mod, a, torch.zeros(()), tie_weights=False)  # tensor(1.)--self.foo_tied is not updated
            >>> new_a = {'foo': torch.zeros(()), 'foo_tied': torch.zeros(())}
            >>> functional_call(mod, new_a, torch.zeros()) # tensor(0.)

    Args:
        module (torch.nn.Module): the module to call
        parameters_and_buffers (dict of str and Tensor): the parameters that will be used in
            the module call.
        args (Any or tuple): arguments to be passed to the module call. If not a tuple, considered a single argument.
        kwargs (dict): keyword arguments to be passed to the module call
        tie_weights (bool, optional): If True, then parameters and buffers tied in the original model will be treated as
            tied in the reparamaterized version. Therefore, if True and different values are passed for the tied
            parameters and buffers, it will error. If False, it will not respect the originally tied parameters and
            buffers unless the values passed for both weights are the same. Default: True.
        strict (bool, optional): If True, then the parameters and buffers passed in must match the parameters and
            buffers in the original module. Therefore, if True and there are any missing or unexpected keys, it will
            error. Default: False.

    Returns:
        Any: the result of calling ``module``.
    """
    return _functional_call(
        module,
        parameters_and_buffers,
        args,
        kwargs,
        tie_weights=tie_weights,
        strict=strict,
    )


def _functional_call(
    module: "torch.nn.Module",
    parameters_and_buffers: dict[str, Tensor],
    args: Any | tuple | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
):
    # TODO allow kwargs such as unsafe and others for parametrization
    if (
        torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or isinstance(
            module,
            (
                torch.jit.RecursiveScriptModule,
                torch.jit.ScriptModule,
                torch.jit.ScriptFunction,
            ),
        )
    ):
        raise RuntimeError("The stateless API can't be used with Jitted modules")
    if isinstance(module, torch.nn.DataParallel):
        raise RuntimeError(
            "The stateless API can't be used with nn.DataParallel module"
        )
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)
    with _reparametrize_module(
        module, parameters_and_buffers, tie_weights=tie_weights, strict=strict
    ):
        return module(*args, **kwargs)
