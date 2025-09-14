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

def sample_cosine_weighted_hemisphere(normal=np.array([0.0, 0.0, 1.0])):
    """
    Cosine-weighted random sampling on a hemisphere oriented around 'normal'.

    Parameters:
        normal: np.ndarray
            The surface normal (3D unit vector). Default is +Z axis.

    Returns:
        np.ndarray: Random light direction (3D unit vector).
    """
    # 1. 두 개의 uniform random numbers
    u1 = np.random.rand()
    u2 = np.random.rand()

    # 2. cosine-weighted hemisphere 샘플링 (local space, z=normal)
    r = np.sqrt(u1)
    theta = 2 * np.pi * u2

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(max(0.0, 1.0 - u1))  # 항상 hemisphere의 위쪽

    local_dir = np.array([x, y, z])

    # 3. normal 방향에 맞게 회전 (from [0,0,1] to normal)
    normal = normal / np.linalg.norm(normal)
    if abs(normal[2]) < 0.999:  
        # 법선과 z축이 평행하지 않을 경우
        tangent = np.cross(np.array([0.0, 0.0, 1.0]), normal)
        tangent /= np.linalg.norm(tangent)
    else:
        tangent = np.array([1.0, 0.0, 0.0])
    bitangent = np.cross(normal, tangent)

    # Local → World transform
    world_dir = (
        local_dir[0] * tangent +
        local_dir[1] * bitangent +
        local_dir[2] * normal
    )
    return world_dir / np.linalg.norm(world_dir)