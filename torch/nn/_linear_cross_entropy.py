import torch


def ensure_size(input, dim, size):
    if input.shape[dim] != size:
        return input.narrow(dim, 0, size)
    return input


def chunk_iter(total_size, chunk_size):
    for start in range(0, total_size, chunk_size):
        if start + chunk_size > total_size:
            yield start, total_size - start
        else:
            yield start, chunk_size


def linear_cross_entropy_batch_chunking_cls_setup_context(ctx, inputs, output):
    ctx.grad_inplace, ctx.compute_input_grad, ctx.compute_linear_weight_grad = inputs[
        -3:
    ]
    _, grad_input, grad_linear_weight = output
    save_indices: list[int | None] = [None, None]
    saved = []
    if ctx.compute_input_grad:
        save_indices[0] = len(saved)
        saved.append(grad_input)
    if ctx.compute_linear_weight_grad:
        save_indices[1] = len(saved)
        saved.append(grad_linear_weight)
    if saved:
        ctx.save_indices = save_indices
        ctx.save_for_backward(*saved)


@torch.library.custom_op(
    "torch_nn::linear_cross_entropy_batch_chunking_cls", mutates_args=()
)
def linear_cross_entropy_batch_chunking_cls(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
    grad_inplace: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = input.device
    dtype = input.dtype
    num_batches, in_features = input.shape
    num_classes, _ = linear_weight.shape

    if dtype != acc_dtype and not (
        dtype in {torch.float16, torch.bfloat16}
        and acc_dtype == torch.float32
        and input.is_cuda
    ):
        raise RuntimeError(
            "linear_cross_entropy_batch_chunking_cls supports float32 acc_dtype on a CUDA device with"
            f" float16/bfloat16 inputs, but got {acc_dtype} acc_type and {dtype} inputs on a {device.type.upper()} device."
        )
    use_acc_dtype = dtype != acc_dtype and input.is_cuda

    if target.dtype != torch.int64:
        raise TypeError(
            f"linear_cross_entropy_batch_chunking_cls: target dtype must be torch.int64, got {target.dtype}."
        )
    mask = target == ignore_index
    if ignore_index < 0 or ignore_index >= num_classes:
        # map out-of-range ignore_index to 0:
        target = torch.where(mask, 0, target)
        # The correctness of this mapping is subtle: mask contains the
        # original target that ensures that selected weights are
        # masked from out-of-range ignore_index mapping correctly.
    neg_weight_target = torch.where(mask, 0, weight.index_select(0, target))

    if reduction == "mean":
        d = neg_weight_target.sum()
        if d == 0:
            # ensure compatibility with cross_entropy
            neg_weight_target.fill_(torch.nan)
        else:
            neg_weight_target.div_(-d)
    elif reduction == "sum":
        neg_weight_target.neg_()
    else:
        raise NotImplementedError(
            f"linear_cross_entropy_batch_chunking_cls does not support {reduction=}"
        )

    if label_smoothing > 0.0:
        raise NotImplementedError(
            "linear_cross_entropy_batch_chunking_cls does not support label smoothing"
        )

    if use_acc_dtype:
        m = dict(T=dtype, A=acc_dtype)
        dtypes = dict(
            output=m[acc_policy[0]],
            grad_input=m[acc_policy[1]],
            grad_linear_weight=m[acc_policy[2]],
            X=m[acc_policy[3]],
            G=m["A" if "A" in acc_policy[4:] else "T"],
            GX=m[acc_policy[4]],
            GL=m[acc_policy[5]],
        )
    else:
        dtypes = dict(
            output=dtype,
            grad_input=dtype,
            grad_linear_weight=dtype,
            X=dtype,
            G=dtype,
            GX=dtype,
            GL=dtype,
        )

    # A chunk buffer used to hold logits, softmax of logits:

    X = torch.empty(
        (batch_chunk_size, num_classes),
        device=device,
        dtype=dtypes["X"],
        requires_grad=False,
    )

    if compute_input_grad and use_acc_dtype:
        linear_weight_ = linear_weight.to(X.dtype)
    else:
        linear_weight_ = linear_weight

    grad_input_shape = input.shape if compute_input_grad else (0,)
    grad_linear_weight_shape = (
        linear_weight.shape if compute_linear_weight_grad else (0,)
    )

    grad_input = torch.zeros(  # TODO: use empty?
        grad_input_shape, dtype=dtypes["grad_input"], device=device, requires_grad=False
    )
    grad_linear_weight = torch.zeros(
        grad_linear_weight_shape,
        dtype=dtypes["grad_linear_weight"],
        device=device,
        requires_grad=False,
    )

    # Allocate junk buffers used in grad_input and grad_linear_weight computations:
    GX_factor = dtypes["G"].itemsize // dtypes["GX"].itemsize
    GL_factor = dtypes["G"].itemsize // dtypes["GL"].itemsize
    tmp_size = (
        max(
            batch_chunk_size // GX_factor if compute_input_grad else 0,
            num_classes // GL_factor if compute_linear_weight_grad else 0,
        )
        * in_features
    )
    tmp = torch.empty(tmp_size, device=device, dtype=dtypes["G"], requires_grad=False)
    GX = (
        torch.narrow(tmp.view(dtypes["GX"]), 0, 0, batch_chunk_size * in_features).view(
            (batch_chunk_size, in_features)
        )
        if compute_input_grad
        else tmp
    )
    GL = (
        torch.narrow(tmp.view(dtypes["GL"]), 0, 0, num_classes * in_features).view(
            (num_classes, in_features)
        )
        if compute_linear_weight_grad
        else tmp
    )

    if reduction in {"mean", "sum"}:
        output = torch.zeros(
            (), device=device, dtype=dtypes["output"], requires_grad=False
        )
    else:
        raise NotImplementedError(
            f"linear_cross_entropy_batch_chunking_cls does not support {reduction=}"
        )
    # chunking along batches dimension:
    for bchunk_start, bchunk_size in chunk_iter(num_batches, batch_chunk_size):
        x = input.narrow(0, bchunk_start, bchunk_size)
        x_ = x.to(acc_dtype)
        t = target.narrow(0, bchunk_start, bchunk_size)
        neg_weight_t = neg_weight_target.narrow(0, bchunk_start, bchunk_size)
        neg_weight_t_ = neg_weight_t.to(X.dtype)
        X_ = ensure_size(X, 0, bchunk_size)
        # Compute output.

        if use_acc_dtype:
            torch.mm(x, linear_weight.T, out_dtype=X_.dtype, out=X_)
        else:
            torch.mm(x, linear_weight.T, out=X_)  # projection

        Xmax = X_.max(dim=1, keepdim=True)[0]
        X_.sub_(Xmax)

        output.add_(neg_weight_t_.dot(X_.gather(1, t.unsqueeze(1)).squeeze(1)))

        X_.exp_()

        expXsum = X_.sum(dim=1)

        if compute_input_grad or compute_linear_weight_grad:
            X_.mul_((neg_weight_t_ / expXsum).unsqueeze(1))

        expXsum.log_()
        output.sub_(neg_weight_t_.dot(expXsum))

        # Compute gradients.

        if compute_input_grad or compute_linear_weight_grad:
            if compute_input_grad:
                grad_x = grad_input.narrow(0, bchunk_start, bchunk_size)
                if use_acc_dtype:
                    GX_ = ensure_size(GX, 0, bchunk_size)
                    if X_.dtype != GX.dtype:
                        GX_[:] = torch.mm(X_, linear_weight_)
                    else:
                        torch.mm(X_, linear_weight_, GX_.dtype, out=GX_)
                    if grad_x.dtype == dtype:
                        # addcmul+sub_ and sub_+addcmul lead to different results!
                        # TODO: use the order with better accuracy
                        torch.addcmul(
                            grad_x,
                            torch.index_select(linear_weight, 0, t),
                            neg_weight_t.unsqueeze(1),
                            out=grad_x,
                        )
                        grad_x.sub_(GX_)
                    else:
                        # no accuracy difference between addcmul+sub_ and sub_+addcmul
                        torch.addcmul(
                            grad_x,
                            torch.index_select(linear_weight_, 0, t),
                            neg_weight_t_.unsqueeze(1),
                            out=grad_x,
                        )
                        grad_x.sub_(GX_)
                else:
                    torch.addcmul(
                        grad_x,
                        torch.index_select(linear_weight, 0, t),
                        neg_weight_t.unsqueeze(1),
                        out=grad_x,
                    )
                    torch.addmm(grad_x, X_, linear_weight, alpha=-1, out=grad_x)

            if compute_linear_weight_grad:
                # avoids zero_ call and a new allocation from x_ * neg_weight_t_
                if X.dtype != GL.dtype:
                    if X_.dtype == x.dtype:
                        GL[:] = torch.mm(X_.T, x)
                    else:
                        GL[:] = torch.mm(X_.T, x_)
                else:
                    if X_.dtype == x.dtype:
                        torch.mm(X_.T, x, out=GL)
                    else:
                        torch.mm(X_.T, x_, out=GL)
                if dtype != acc_dtype:
                    # x_ is a copy of input slice, so we can
                    # change it inplace to reduce memory usage
                    x_.mul_(neg_weight_t_.unsqueeze(1))
                    if GL.dtype == x.dtype:
                        GL.index_add_(0, t, x, alpha=-1)
                    else:
                        GL.index_add_(0, t, x_, alpha=-1)
                else:
                    GL.index_add_(0, t, x_ * neg_weight_t_.unsqueeze(1), alpha=-1)
                grad_linear_weight.sub_(GL)

    return output.to(dtype), grad_input.to(dtype), grad_linear_weight.to(dtype)


@linear_cross_entropy_batch_chunking_cls.register_fake
def _(
    input,
    linear_weight,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
    batch_chunk_size,
    acc_policy: str,
    acc_dtype,
    grad_inplace,
    compute_input_grad,
    compute_linear_weight_grad,
):
    if reduction in {"mean", "sum"}:
        result = torch.empty((), dtype=input.dtype, device=input.device)
    else:
        raise NotImplementedError(
            f"linear_cross_entropy_batch_chunking_cls does not support {reduction=}"
        )
    grad_input_shape = input.shape if compute_input_grad else (0,)
    grad_linear_weight_shape = (
        linear_weight.shape if compute_linear_weight_grad else (0,)
    )

    grad_input = torch.empty(
        grad_input_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=False,
    )
    grad_linear_weight = torch.empty(
        grad_linear_weight_shape,
        dtype=linear_weight.dtype,
        device=linear_weight.device,
        requires_grad=False,
    )
    return result, grad_input, grad_linear_weight


def linear_cross_entropy_batch_chunking_cls_backward(ctx, *grads):
    grad_output = grads[0]
    result = [None] * 13

    if ctx.compute_input_grad or ctx.compute_linear_weight_grad:
        saved = ctx.saved_tensors
        if ctx.grad_inplace:
            # With grad_inplace, the memory usage size is reduced
            # 2x when reusing pre-computed grad_input and
            # grad_linear_weight storages. However, gradcheck does
            # not like that.
            if ctx.compute_input_grad:
                grad_input = saved[ctx.save_indices[0]]
                grad_input.mul_(grad_output)
                result[0] = grad_input
            if ctx.compute_linear_weight_grad:
                grad_linear_weight = saved[ctx.save_indices[1]]
                grad_linear_weight.mul_(grad_output)
                result[1] = grad_linear_weight
        else:
            # gradcheck-friendly backward:
            if ctx.compute_input_grad:
                grad_input = saved[ctx.save_indices[0]]
                # creates a new tensor that increases memory usage size
                result[0] = grad_input * grad_output
            if ctx.compute_linear_weight_grad:
                grad_linear_weight = saved[ctx.save_indices[1]]
                # creates a new tensor that increases memory usage size
                result[1] = grad_linear_weight * grad_output

    return tuple(result)


linear_cross_entropy_batch_chunking_cls.register_autograd(
    linear_cross_entropy_batch_chunking_cls_backward,
    setup_context=linear_cross_entropy_batch_chunking_cls_setup_context,
)
