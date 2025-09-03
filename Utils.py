import slangpy as spy

def downsample(device, module, source: spy.Tensor, steps: int) -> spy.Tensor:
    for i in range(steps):
        dest = spy.Tensor.empty(
            device=device,
            shape=(source.shape[0] // 2, source.shape[1] // 2),
            dtype=source.dtype)
        module.downsample(spy.call_id(), source, _result=dest)
        source = dest

    return source