import slangpy as spy
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

device_type = spy.DeviceType.automatic
device = spy.create_device(
    device_type, enable_debug_layers=False, include_paths=[EXAMPLE_DIR]
)

print(device)

module = spy.Module.load_from_file(device, "Simple.slang")

#test loading an image
input = spy.Tensor.load_from_image(device, EXAMPLE_DIR.joinpath("PavingStones070_2K.diffuse.jpg"), linearize=True)

tex = device.create_texture(format=spy.Format.rgba32_float, width=input.shape[1], height=input.shape[0], usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access)
module.copy(input, tex)

spy.tev.show(tex)

module.brighten(amount=spy.float3(0.5), pixel=tex)

spy.tev.show(tex)