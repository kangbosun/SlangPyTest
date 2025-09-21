import slangpy as spy
import numpy as np

def downsample(device, module, source: spy.Tensor, steps: int) -> spy.Tensor:
    for i in range(steps):
        dest = spy.Tensor.empty(
            device=device,
            shape=(source.shape[0] // 2, source.shape[1] // 2),
            dtype=source.dtype)
        module.downsample(spy.call_id(), source, _result=dest)
        source = dest

    return source

def sample_cosine_weighted_hemisphere(epoch, period=100):
    """
    Cosine-weighted random sampling on a hemisphere around fixed normal (0,0,1).
    Sampling is deterministic based on epoch, and repeats every 'period' steps.

    Parameters:
        epoch: int
            Current epoch (used to seed RNG).
        period: int
            Number of epochs before the sequence repeats (default: 100).

    Returns:
        np.ndarray: Deterministic light direction (3D unit vector).
    """
    # 1. deterministic seed
    seed = epoch % period
    rng = np.random.RandomState(seed)

    # 2. cosine-weighted hemisphere sampling (z = up)
    u1 = rng.rand()
    u2 = rng.rand()

    r = np.sqrt(u1)
    theta = 2 * np.pi * u2

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(max(0.0, 1.0 - u1))  # hemisphere 위쪽

    return np.array([x, y, z])