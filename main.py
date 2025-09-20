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
       
        self.light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
        self.ref_output = None

        self.downsample_steps = 2
        spy.ui.InputInt(app.ui_window, "Downsample Steps", value=self.downsample_steps, callback=self.on_downsample_steps_changed)

        self.tex_size_text = spy.ui.Text(app.ui_window, "Miplevel: {0} ({1}x{2})".format(self.downsample_steps, self.albedo_map.shape[1] >> self.downsample_steps, self.albedo_map.shape[0] >> self.downsample_steps))

        spy.ui.Text(app.ui_window, " ")

        self.init_textures(app)

        ui_group_view = spy.ui.Group(app.ui_window, "View")
        self.view_mode = 0
        self.view_combobox = spy.ui.ComboBox(ui_group_view, "View Mode", items=["trained", "gradient", "mean", "variance"], callback=self.on_view_mode_changed)

        self.view_scale = 1
        spy.ui.SliderFloat(ui_group_view, "View Scale", value=self.view_scale, callback=lambda v: setattr(self, 'view_scale', v), min=0.25, max=8.0)

        # self.stretch = False
        # spy.ui.CheckBox(ui_group_view, "Stretch", value=self.stretch, callback=self.on_stretch_changed)

        self.light_dir_drag = spy.ui.DragFloat3(ui_group_view, "Light Dir", value=self.light_dir, min=-1.0, max=1.0, speed=0.05, callback=lambda v: setattr(self, 'light_dir', spy.math.normalize(v)))

        self.metallic = 0.0
        spy.ui.SliderFloat(ui_group_view, "Metallic", value=self.metallic, callback=self.on_metallic_changed, min=0.0, max=1.0)

        spy.ui.Text(app.ui_window, " ")

        ui_group_training = spy.ui.Group(app.ui_window, "Training")
        self.max_epoch = 3000
        self.current_epoch = 0
        spy.ui.InputInt(ui_group_training, "Max epoch", value=self.max_epoch, callback=lambda v: setattr(self, 'max_epoch', v))

        self.learning_rate = 0.002
        spy.ui.InputFloat(ui_group_training, "Learning rate", value=self.learning_rate, callback=lambda v: setattr(self, 'learning_rate', v))

        self.optimizer = 0
        self.optimizer_combobox = spy.ui.ComboBox(ui_group_training, "Optimizer", items=["Adam", "SGD"], callback=lambda v: setattr(self, 'optimizer', v))

        self.is_training = False
        self.train_button = spy.ui.Button(ui_group_training, "Train", callback=self.on_train_clicked)

        self.progressbar = spy.ui.ProgressBar(ui_group_training, fraction=0.0)

        spy.ui.Button(ui_group_training, "Reset", callback=self.on_reset_clicked)

    def init_textures(self, app: App):
        self.downsampled_albedo = self.downsample(self.albedo_map, self.downsample_steps)
        self.downsampled_normal = self.downsample(self.normal_map, self.downsample_steps)
        self.downsampled_roughness = self.downsample(self.roughness_map, self.downsample_steps)

        self.trained_albedo = spy.Tensor.empty_like(self.downsampled_albedo)
        self.trained_normal = spy.Tensor.empty_like(self.downsampled_normal)
        self.trained_roughness = spy.Tensor.empty_like(self.downsampled_roughness)

        self.mipmap_module.init3(self.trained_albedo, spy.float3(0.5, 0.5, 0.5))
        self.mipmap_module.init_normal(self.trained_normal)
        self.mipmap_module.init1(self.trained_roughness, 0.5)

        self.albedo_grad = spy.Tensor.empty_like(self.downsampled_albedo)
        self.normal_grad = spy.Tensor.empty_like(self.downsampled_normal)
        self.roughness_grad = spy.Tensor.empty_like(self.downsampled_roughness)

        self.albedo_mean = spy.Tensor.zeros_like(self.downsampled_albedo)
        self.albedo_variance = spy.Tensor.zeros_like(self.downsampled_albedo)
        self.normal_mean = spy.Tensor.zeros_like(self.downsampled_normal)
        self.normal_variance = spy.Tensor.zeros_like(self.downsampled_normal)
        self.roughness_mean = spy.Tensor.zeros_like(self.downsampled_roughness)
        self.roughness_variance = spy.Tensor.zeros_like(self.downsampled_roughness)

    def on_downsample_steps_changed(self, value: int):
        self.downsample_steps = value
        self.init_textures(app)
        self.ref_output = None
        self.on_reset_clicked()
        self.tex_size_text.text = "Miplevel: {0} ({1}x{2})".format(self.downsample_steps, self.albedo_map.shape[1] >> self.downsample_steps, self.albedo_map.shape[0] >> self.downsample_steps)

    def on_stretch_changed(self, value: bool):
        self.stretch = value

    def on_metallic_changed(self, value: float):
        self.metallic = value
        self.ref_output = None

    def on_view_mode_changed(self, value: str):
        self.view_mode = value
        print(self.view_mode)

    def set_light_dir(self, value: spy.float3):
        self.light_dir = spy.math.normalize(value)
        self.light_dir_drag.value = self.light_dir

    def on_reset_clicked(self):
        self.mipmap_module.init3(self.trained_albedo, spy.float3(0.5, 0.5, 0.5))
        self.mipmap_module.init_normal(self.trained_normal)
        self.mipmap_module.init1(self.trained_roughness, 0.5)

        # clear gradients
        self.mipmap_module.init3(self.albedo_grad, spy.float3(0.0, 0.0, 0.0))
        self.mipmap_module.init3(self.normal_grad, spy.float3(0.0, 0.0, 0.0))
        self.mipmap_module.init1(self.roughness_grad, 0.0)

        # clear mean and variance
        self.mipmap_module.init3(self.albedo_mean, spy.float3(0.0, 0.0, 0.0))
        self.mipmap_module.init3(self.normal_mean, spy.float3(0.0, 0.0, 0.0))
        self.mipmap_module.init1(self.roughness_mean, 0.0)
        self.mipmap_module.init3(self.albedo_variance, spy.float3(0.0, 0.0, 0.0))
        self.mipmap_module.init3(self.normal_variance, spy.float3(0.0, 0.0, 0.0))
        self.mipmap_module.init1(self.roughness_variance, 0.0)

        self.set_light_dir(spy.float3(0.2, 0.2, 1.0))

        self.current_epoch = 0
        self.progressbar.fraction = 0.0
        self.pause_train()

    def pause_train(self):
        self.train_button.label = "Train"
        self.is_training = False
        print("Training paused.")

    def resume_train(self):
        can_start = self.current_epoch < self.max_epoch
        if can_start:
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

        org_size = spy.int2(source.shape[1], source.shape[0])

        target_size = size if size is None else size

        scale = spy.float2(org_size.x / target_size.x, org_size.y / target_size.y)

        if offset is None:
            offset = spy.int2(0, 0)

        self.app_module.blit(
            spy.grid((target_size.x, target_size.y)), target_size, offset, tonemap, bilinear, source, output
        )

    def pre_render(self, app):
        return super().pre_render(app)

    def render_reference(self, app: App, light_dir: spy.float3):
        width = self.albedo_map.shape[1]
        height = self.albedo_map.shape[0]

        ref_output = spy.Tensor.empty(app.device, (width, height), 'float3')

        self.mipmap_module.render(pixel=spy.call_id(),
            material = {
                "albedo": self.albedo_map,
                "normal": self.normal_map,
                "roughness": self.roughness_map,
                "metallic": self.metallic
            },
            light_dir=light_dir,
            view_dir=spy.float3(0, 0, 1),

            _result=ref_output
            )
        
        ref_output = self.downsample(ref_output, self.downsample_steps)
        return ref_output
    
    def train(self, app: App, light_dir: spy.float3):
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
        )
    
        # optimize
        if self.optimizer == 0: # Adam
            self.mipmap_module.optimizer_adam3(self.trained_albedo, self.albedo_grad, self.albedo_mean, self.albedo_variance, self.learning_rate, self.current_epoch, False)
            self.mipmap_module.optimizer_adam3(self.trained_normal, self.normal_grad, self.normal_mean, self.normal_variance, self.learning_rate, self.current_epoch, True)
            self.mipmap_module.optimizer_adam1(self.trained_roughness, self.roughness_grad, self.roughness_mean, self.roughness_variance, self.learning_rate, self.current_epoch)
        else: # SGD
            self.mipmap_module.optimizer_sgd3(self.trained_albedo, self.albedo_grad, self.learning_rate, False)
            self.mipmap_module.optimizer_sgd3(self.trained_normal, self.normal_grad, self.learning_rate, True)
            self.mipmap_module.optimizer_sgd1(self.trained_roughness, self.roughness_grad, self.learning_rate)

        self.progressbar.fraction = self.current_epoch / self.max_epoch

        if self.current_epoch >= self.max_epoch:
            self.pause_train()
            self.set_light_dir(spy.float3(0.2, 0.2, 1.0))
            print("Training completed.")
    
    def tick(self, app) :
        should_run_training = self.is_training and self.current_epoch < self.max_epoch
        #create random light direction with numpy
        if should_run_training :
            light_dir_np = Utils.sample_cosine_weighted_hemisphere()
            self.set_light_dir(spy.float3(light_dir_np[0], light_dir_np[1], light_dir_np[2]))
            self.ref_output = self.render_reference(app, self.light_dir)

            self.train(app, self.light_dir)

    def render(self, app: App):
        self.ref_output = self.render_reference(app, self.light_dir)

        self.mipmap_module.clear(spy.float4(0.0), app.output_texture)

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
            light_dir=self.light_dir,
            view_dir=spy.float3(0, 0, 1),
            _result=rendered_output
            )


        out_size = spy.int2((int)(self.ref_output.shape[0] * self.view_scale), (int)(self.ref_output.shape[1] * self.view_scale))
        # show reference

        self.blit(self.ref_output, app.output_texture, size=out_size, offset=spy.int2(0, 0), tonemap=True, bilinear=True)
        xpos = out_size.x + 10
        ypos = 0

        self.blit(rendered_output, app.output_texture, size=out_size, offset=spy.int2(xpos, 0), tonemap=True, bilinear=True)  
        xpos = 0
        ypos = out_size.y + 10

        if self.view_mode == 0: # trained
            self.blit(self.trained_albedo, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += out_size.x + 10
            self.blit(self.trained_normal, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += out_size.x + 10
            #self.blit(self.roughness_grad, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)

        elif self.view_mode == 1: # gradient
            self.blit(self.albedo_grad, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += out_size.x + 10
            self.blit(self.normal_grad, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += out_size.x + 10
            #self.blit(self.trained_roughness, app.output_texture, size=spy.int2(self.downsampled_roughness.shape[0], self.downsampled_roughness.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
        elif self.view_mode == 2: # mean
            self.blit(self.albedo_mean, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += out_size.x + 10
            self.blit(self.normal_mean, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += out_size.x + 10
            #self.blit(self.roughness_mean, app.output_texture, size=spy.int2(self.downsampled_roughness.shape[0], self.downsampled_roughness.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
        elif self.view_mode == 3: # variance
            self.blit(self.albedo_variance, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += out_size.x + 10
            self.blit(self.normal_variance, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)
            xpos += out_size.x + 10
            #self.blit(self.roughness_variance, app.output_texture, size=spy.int2(self.downsampled_roughness.shape[0], self.downsampled_roughness.shape[1]), offset=spy.int2(xpos, ypos), tonemap=False, bilinear=True)

        diff_output = spy.Tensor.empty(app.device, (width, height), 'float3')
        self.app_module.diff(self.ref_output, rendered_output, _result=diff_output)
        self.blit(diff_output, app.output_texture, size=out_size, offset=spy.int2(xpos, 0), tonemap=True, bilinear=True)

        return super().render(app)

    def post_render(self, app):
        return super().post_render(app)
    


app = App()
renderer = MipmapRenderer(app)
app.set_renderer(renderer)



app.run()
