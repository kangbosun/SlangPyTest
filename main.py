## main_scalar.py

import os
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

        self.mipmap_module.init3(self.trained_albedo, spy.float3(0.5, 0.5, 0.5))
        self.mipmap_module.init_normal(self.trained_normal)
        self.mipmap_module.init1(self.trained_roughness, 0.5)

        self.albedo_grad = spy.Tensor.empty_like(self.downsampled_albedo)
        self.normal_grad = spy.Tensor.empty_like(self.downsampled_normal)
        self.roughness_grad = spy.Tensor.empty_like(self.downsampled_roughness)

        self.ref_output = None

        self.downsample_steps = 2
        # spy.ui.InputInt(app.ui_window, "Downsample Steps", value=self.downsample_steps, callback=self.on_downsample_steps_changed)

        self.view_mode = 0
        spy.ui.ComboBox(app.ui_window, "View Mode", items=["trained", "gradient"], callback=self.on_view_mode_changed)

        self.stretch = False
        spy.ui.CheckBox(app.ui_window, "Stretch", value=self.stretch, callback=self.on_stretch_changed)

        self.metallic = 0.0
        spy.ui.SliderFloat(app.ui_window, "Metallic", value=self.metallic, callback=self.on_metallic_changed, min=0.0, max=1.0)
      
        ui_group_training = spy.ui.Group(app.ui_window, "Training")
        self.max_epoch = 300
        self.current_epoch = 0
        spy.ui.InputInt(ui_group_training, "Max epoch", value=self.max_epoch, callback=lambda v: setattr(self, 'max_epoch', v))

        self.learning_rate = 0.05
        spy.ui.InputFloat(ui_group_training, "Learning rate", value=self.learning_rate, callback=lambda v: setattr(self, 'learning_rate', v))

        self.is_training = False
        self.train_button = spy.ui.Button(ui_group_training, "Train", callback=self.on_train_clicked)

        self.progressbar = spy.ui.ProgressBar(ui_group_training, fraction=0.0)

        spy.ui.Button(ui_group_training, "Reset", callback=self.on_reset_clicked)

    def on_downsample_steps_changed(self, value: int):
        #clamp 0 to 5
        self.downsample_steps = max(0, min(5, value))

    def on_stretch_changed(self, value: bool):
        self.stretch = value

    def on_metallic_changed(self, value: float):
        self.metallic = value
        self.ref_output = None

    def on_view_mode_changed(self, value: str):
        self.view_mode = value
        print(self.view_mode)

    def on_reset_clicked(self):
        self.mipmap_module.init3(self.trained_albedo, spy.float3(0.5, 0.5, 0.5))
        self.mipmap_module.init_normal(self.trained_normal)
        self.mipmap_module.init1(self.trained_roughness, 0.5)

        # clear gradients
        self.mipmap_module.init3(self.albedo_grad, spy.float3(0.0, 0.0, 0.0))
        self.mipmap_module.init3(self.normal_grad, spy.float3(0.0, 0.0, 0.0))
        self.mipmap_module.init1(self.roughness_grad, 0.0)
        
        self.current_epoch = 0
        self.progressbar.fraction = 0.0
        self.pause_train()

    def pause_train(self):
        self.train_button.label = "Train"
        self.is_training = False
        print("Training paused.")

    def resume_train(self):
        self.train_button.label = "Pause"
        self.is_training = True
        print("Training resumed.")

    def on_train_clicked(self):
        if self.is_training:
            self.pause_train()
        else:
            self.resume_train()

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

    def render_reference(self, app: App, light_dir: spy.float3):
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
            light_dir=light_dir,
            view_dir=spy.float3(0, 0, 1),
            view_scale=view_scale,
            _result=ref_output
            )
        
        ref_output = self.downsample(ref_output, self.downsample_steps)
        return ref_output
    
    def train(self, app: App, light_dir: spy.float3, view_scale: float):
        if self.current_epoch < self.max_epoch:
            #print(f"Training epoch {self.current_epoch+1}/{self.max_epoch}")
            self.current_epoch += 1
            self.mipmap_module.calculate_grad(0, 
            pixel=spy.call_id(),
            reference=self.ref_output,
            material={
                "albedo": self.trained_albedo,
                "normal": self.trained_normal,
                "roughness": self.trained_roughness,
                "metallic": self.metallic,
                "albedo_grad": self.albedo_grad,
                "normal_grad": self.normal_grad,
                "roughness_grad": self.roughness_grad,
            },
            light_dir=light_dir,
            view_dir=spy.float3(0, 0, 1),
            view_scale=view_scale
            )
        
            # optimize
            self.mipmap_module.optimizer_step3(self.trained_albedo, self.albedo_grad, self.learning_rate)
            self.mipmap_module.optimizer_step3(self.trained_normal, self.normal_grad, self.learning_rate)
            self.mipmap_module.optimizer_step1(self.trained_roughness, self.roughness_grad, self.learning_rate)

            self.progressbar.fraction = self.current_epoch / self.max_epoch

            if self.current_epoch >= self.max_epoch:
                self.pause_train()
                print("Training completed.")
        else:
            self.pause_train()
    
    def render(self, app: App):
        self.mipmap_module.clear(spy.float4(0.0), app.output_texture)
        
        width = self.albedo_map.shape[1]
        height = self.albedo_map.shape[0]

        view_scale = 1.0
        light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))

        #create random light direction with numpy
        if self.is_training:
            light_dir_np = Utils.sample_cosine_weighted_hemisphere()
            light_dir = spy.math.normalize(spy.float3(light_dir_np[0], light_dir_np[1], light_dir_np[2]))

        #if self.ref_output is None:
        self.ref_output = self.render_reference(app, light_dir)

        if self.is_training:
            self.train(app, light_dir, view_scale)

        width = self.trained_albedo.shape[1]
        height = self.trained_albedo.shape[0]

        rendered_output = spy.Tensor.empty(app.device, (width, height), 'float3')
        self.mipmap_module.render(pixel=spy.call_id(),
            material = {
                "albedo": self.trained_albedo,
                "normal": self.trained_normal,
                "roughness": self.trained_roughness,
                "metallic": self.metallic
            },
            light_dir=light_dir,
            view_dir=spy.float3(0, 0, 1),
            view_scale=view_scale,
            _result=rendered_output
            )

        # show reference
        self.blit(self.ref_output, app.output_texture, size=spy.int2(self.ref_output.shape[0], self.ref_output.shape[1]), offset=spy.int2(0, 0), tonemap=True, bilinear=True)
        xpos = self.ref_output.shape[0] + 10
        ypos = 0

        self.blit(rendered_output, app.output_texture, size=spy.int2(rendered_output.shape[0], rendered_output.shape[1]), offset=spy.int2(xpos, 0), tonemap=True, bilinear=True)  
        xpos = 0
        ypos = self.ref_output.shape[1] + 10

        if self.view_mode == 0:
            self.blit(self.trained_albedo, app.output_texture, size=spy.int2(self.downsampled_albedo.shape[0], self.downsampled_albedo.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += self.trained_albedo.shape[0] + 10
            self.blit(self.trained_normal, app.output_texture, size=spy.int2(self.downsampled_normal.shape[0], self.downsampled_normal.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += self.trained_normal.shape[0] + 10
            #self.blit(self.roughness_grad, app.output_texture, size=spy.int2(self.downsampled_roughness.shape[0], self.downsampled_roughness.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)

        elif self.view_mode == 1:
            self.blit(self.albedo_grad, app.output_texture, size=spy.int2(self.downsampled_albedo.shape[0], self.downsampled_albedo.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += self.albedo_grad.shape[0] + 10
            self.blit(self.normal_grad, app.output_texture, size=spy.int2(self.downsampled_normal.shape[0], self.downsampled_normal.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += self.normal_grad.shape[0] + 10
            #self.blit(self.trained_roughness, app.output_texture, size=spy.int2(self.downsampled_roughness.shape[0], self.downsampled_roughness.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)

        diff_output = spy.Tensor.empty(app.device, (width, height), 'float3')
        self.app_module.diff(self.ref_output, rendered_output, _result=diff_output)
        self.blit(diff_output, app.output_texture, size=spy.int2(diff_output.shape[0], diff_output.shape[1]), offset=spy.int2(xpos, 0), tonemap=True, bilinear=True)

        return super().render(app)

    def post_render(self, app):
        return super().post_render(app)
    


app = App()
renderer = MipmapRenderer(app)
app.set_renderer(renderer)



app.run()
