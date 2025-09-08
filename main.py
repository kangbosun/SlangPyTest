## main_scalar.py

import slangpy as spy
import numpy as np
import pathlib
from pathlib import Path
import Utils
from App import App, Renderer

class MipmapRenderer(Renderer):
    def __init__(self, app: App):

        data_path = Path(__file__).parent

        self.mipmap_module = spy.Module.load_from_file(app.device, "Mipmap.slang")
        self.app_module = spy.Module.load_from_file(app.device, "App.slang")

        # Load some materials.
        self.albedo_map = spy.Tensor.load_from_image(
            app.device, data_path.joinpath("PavingStones070_2K.diffuse.jpg"), linearize=True
        )
        self.normal_map = spy.Tensor.load_from_image(
            app.device, data_path.joinpath("PavingStones070_2K.normal.jpg"), scale=2, offset=-1
        )
        self.roughness_map = spy.Tensor.load_from_image(
            app.device, data_path.joinpath("PavingStones070_2K.roughness.jpg"), grayscale=True
        )

        self.downsampled_albedo = self.downsample(self.albedo_map, 2)
        self.downsampled_normal = self.downsample(self.normal_map, 2)
        self.downsampled_roughness = self.downsample(self.roughness_map, 2)

        self.trained_albedo = spy.Tensor.empty_like(self.downsampled_albedo)
        self.trained_normal = spy.Tensor.empty_like(self.downsampled_normal)
        self.trained_roughness = spy.Tensor.empty_like(self.downsampled_roughness)

        self.albedo_grad = spy.Tensor.empty_like(self.downsampled_albedo)
        self.normal_grad = spy.Tensor.empty_like(self.downsampled_normal)
        self.roughness_grad = spy.Tensor.empty_like(self.downsampled_roughness)

        self.downsample_steps = 2
        # spy.ui.InputInt(app.ui_window, "Downsample Steps", value=self.downsample_steps, callback=self.on_downsample_steps_changed)

        self.material_mode = 0
        spy.ui.ComboBox(app.ui_window, "Material Mode", items=["all", "Reference", "Real Render", "Loss"], callback=self.on_material_changed)

        self.stretch = False
        spy.ui.CheckBox(app.ui_window, "Stretch", value=self.stretch, callback=self.on_stretch_changed)

        self.metallic = 0.0
        spy.ui.SliderFloat(app.ui_window, "Metallic", value=self.metallic, callback=self.on_metallic_changed, min=0.0, max=1.0)

    def on_downsample_steps_changed(self, value: int):
        #clamp 0 to 5
        self.downsample_steps = max(0, min(5, value))

    def on_stretch_changed(self, value: bool):
        self.stretch = value

    def on_metallic_changed(self, value: float):
        self.metallic = value

    def on_material_changed(self, value: str):
        self.material_mode = value
        print(self.material_mode)

    def downsample(self, source: spy.Tensor, steps: int) -> spy.Tensor:
        for i in range(steps):
            dest = spy.Tensor.empty(
                device=app.device,
                shape=(source.shape[0] // 2, source.shape[1] // 2),
                dtype=source.dtype,
            )
            if dest.dtype.name == "vector":
                self.mipmap_module.downsample3(spy.call_id(), source, _result=dest)
            else:
                self.mipmap_module.downsample1(spy.call_id(), source, _result=dest)
            source = dest
        return source

    def blit(self, source: spy.Tensor, output:spy.Texture, size: spy.int2 = None, offset: spy.int2 = None, tonemap: bool = True, bilinear: bool = False):
        if len(source.shape) != 2:
            raise ValueError("Source tensor must be 2D (height, width).")
        if size is None:
            size = spy.int2(source.shape[1], source.shape[0])
        if offset is None:
            offset = spy.int2(0, 0)

        self.app_module.blit(
            spy.grid((size.y, size.x)), size, offset, tonemap, bilinear, source, output
        )

    def pre_render(self, app):
        return super().pre_render(app)
    
    def render(self, app: App):
        self.mipmap_module.clear(spy.float4(0.0), app.output_texture)
        
        
        width = self.albedo_map.shape[1]
        height = self.albedo_map.shape[0]

        ref_output = spy.Tensor.empty(app.device, (width, height), 'float3')
        view_scale = 1.0
        self.mipmap_module.render(pixel=spy.call_id(),
            material = {
                "albedo": self.albedo_map,
                "normal": self.normal_map,
                "roughness": self.roughness_map,
                "metallic": self.metallic
            },
            light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
            view_dir=spy.float3(0, 0, 1),
            view_scale=view_scale,
            _result=ref_output
            )
        
        ref_output = self.downsample(ref_output, self.downsample_steps)

        downscaled_width = self.albedo_map.shape[1] // (2 ** self.downsample_steps)
        downscaled_height = self.albedo_map.shape[0] // (2 ** self.downsample_steps)

        real_output = spy.Tensor.empty(app.device, (downscaled_width, downscaled_height), 'float3')

        view_scale = 1.0
        self.mipmap_module.render(pixel=spy.call_id(),
            material = {
                "albedo": self.downsampled_albedo,
                "normal": self.downsampled_normal,
                "roughness": self.downsampled_roughness,
                "metallic": self.metallic
            },
            light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
            view_dir=spy.float3(0, 0, 1),
            view_scale=view_scale,
            _result=real_output
            )


        loss_output = spy.Tensor.empty_like(real_output)
        self.mipmap_module.loss(ref_output, real_output, _result=loss_output)

        

        if self.material_mode == 0: # all
            ypos = 0
            xpos = 0
            self.blit(ref_output, app.output_texture, size=spy.int2(ref_output.shape[0], ref_output.shape[1]), tonemap=True, bilinear=True)
            xpos += ref_output.shape[1] + 10
            self.blit(real_output, app.output_texture, size=spy.int2(real_output.shape[0], real_output.shape[1]), offset=spy.int2(xpos, ypos), tonemap=True, bilinear=True)
            xpos += real_output.shape[1] + 10
            self.blit(loss_output, app.output_texture, size=spy.int2(loss_output.shape[0], loss_output.shape[1]), offset=spy.int2(xpos, ypos), tonemap=True, bilinear=True)
        elif self.material_mode == 1: # Reference
            size = spy.int2(app.output_texture.width, app.output_texture.height) if self.stretch else spy.int2(ref_output.shape[0], ref_output.shape[1])
            self.blit(ref_output, app.output_texture, size=size, tonemap=True, bilinear=True)
        elif self.material_mode == 2: # Real
            size = spy.int2(app.output_texture.width, app.output_texture.height) if self.stretch else spy.int2(real_output.shape[0], real_output.shape[1])
            self.blit(real_output, app.output_texture, size=size, tonemap=True, bilinear=True)
        elif self.material_mode == 3: # Loss
            size = spy.int2(app.output_texture.width, app.output_texture.height) if self.stretch else spy.int2(loss_output.shape[0], loss_output.shape[1])
            self.blit(loss_output, app.output_texture, size=size, tonemap=True, bilinear=True)

        return super().render(app)

    def post_render(self, app):
        return super().post_render(app)
    


width = 512
height = 512

app = App()
renderer = MipmapRenderer(app)
app.set_renderer(renderer)



app.run()
