## main_scalar.py

import os
import io
import slangpy as spy
import numpy as np
import pathlib
from pathlib import Path
import Utils
from App import App, Renderer
import matplotlib.pyplot as plt

VIEW_SIZE_SCALE = 1.5  # Scaling factor for output size

# Global loss history struct for visualization key is epoch, value is loss
class LossHistory:
    def __init__(self):
        self.history = []

    def add(self, loss, epoch):
        self.history.append((epoch, loss))

    def clear(self):
        self.history = []
        plt.clf()

    #visualize to memory
    def visualize(self):
        
        epochs, losses = zip(*self.history)

        plt.ion()
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
                
        #plt.show(block=False)
        plt.pause(0.01)

    def stop_visualize(self):
        plt.ioff() 
# Vertical spacing constant for layout
VERTICAL_SPACING = 50

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

        self.prev_mouse_pos = spy.float2()
        self.view_offset = spy.int2(0, 0)
       
        self.light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
        self.ref_output = None

        self.downsample_steps = 2
        spy.ui.InputInt(app.ui_window, "Downsample Steps", value=self.downsample_steps, callback=self.on_downsample_steps_changed)

        self.tex_size_text = spy.ui.Text(app.ui_window, "Miplevel: {0} ({1}x{2})".format(self.downsample_steps, self.albedo_map.shape[1] >> self.downsample_steps, self.albedo_map.shape[0] >> self.downsample_steps))

        spy.ui.Text(app.ui_window, " ")

        self.init_textures(app)

        self.view_offset = spy.int2(0, 0)

        ui_group_view = spy.ui.Group(app.ui_window, "View")
        self.view_mode = 0

        self.view_combobox = spy.ui.ComboBox(ui_group_view, "Compare to", items=["Rendered->Downsampled", "Downsampled->Rendered"], callback=self.on_view_mode_changed)

        self.view_scale = 1
        spy.ui.SliderFloat(ui_group_view, "View Scale", value=self.view_scale, callback=lambda v: setattr(self, 'view_scale', v), min=0.25, max=8.0)

        # self.stretch = False
        # spy.ui.CheckBox(ui_group_view, "Stretch", value=self.stretch, callback=self.on_stretch_changed)

        self.light_dir_drag = spy.ui.DragFloat3(ui_group_view, "Light Dir", value=self.light_dir, min=-1.0, max=1.0, speed=0.05, callback=self.on_light_dir_changed)

        self.metallic = 0.0
        spy.ui.SliderFloat(ui_group_view, "Metallic", value=self.metallic, callback=self.on_metallic_changed, min=0.0, max=1.0)

        spy.ui.Text(app.ui_window, " ")

        ui_group_training = spy.ui.Group(app.ui_window, "Training")
        self.max_epoch = 30
        self.current_epoch = 1
        spy.ui.InputInt(ui_group_training, "Max epoch", value=self.max_epoch, callback=lambda v: setattr(self, 'max_epoch', v))

        self.learning_rate = 0.002
        spy.ui.InputFloat(ui_group_training, "Learning rate", value=self.learning_rate, callback=lambda v: setattr(self, 'learning_rate', v))

        self.optimizer = 0
        self.optimizer_combobox = spy.ui.ComboBox(ui_group_training, "Optimizer", items=["Adam", "SGD"], callback=lambda v: setattr(self, 'optimizer', v))

        self.is_training = False
        self.train_button = spy.ui.Button(ui_group_training, "Train", callback=self.on_train_clicked)

        self.progressbar = spy.ui.ProgressBar(ui_group_training, fraction=0.0)

        self.avg_loss = 0.0
        self.loss_format = "Loss avg: {0:.4f}| max: {1:.4f}"
        self.loss_text = spy.ui.Text(ui_group_training, self.loss_format.format(0.0, 0.0))

        self.loss_history = LossHistory()

        PERIOD = 100
        self.presampled_light_dirs = []
        for i in range(100):
            light_dir_np = Utils.sample_cosine_weighted_hemisphere(i, period=PERIOD)
            self.presampled_light_dirs.append(spy.float3(light_dir_np[0], light_dir_np[1], light_dir_np[2]))

        self.current_light_dir_index = 0

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
        self.rendered_output = None

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
        self.on_light_dir_changed(value)
        self.light_dir_drag.value = self.light_dir

    def on_light_dir_changed(self, value: spy.float3):
        self.light_dir = spy.math.normalize(value)
        self.ref_output = None

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

        self.current_epoch = 1
        self.progressbar.fraction = 0.0

        self.loss_history.clear()
        self.loss_text.text = self.loss_format.format(0.0, 0.0)
        self.pause_train()

    def pause_train(self):
        self.train_button.label = "Train"
        self.is_training = False
        self.loss_history.stop_visualize()
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
        
        target_size = size if size is None else size

        if offset is None:
            offset = spy.int2(0, 0)

        self.app_module.blit(
            spy.grid((target_size.x, target_size.y)), target_size, offset, tonemap, bilinear, source, output
        )

    def blit1(self, source: spy.Tensor, output:spy.Texture, size: spy.int2 = None, offset: spy.int2 = None, tonemap: bool = True, bilinear: bool = False):
        if len(source.shape) != 2:
            raise ValueError("Source tensor must be 2D (height, width).")

        target_size = size if size is None else size

        if offset is None:
            offset = spy.int2(0, 0)

        self.app_module.blit1(
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
        width = self.trained_albedo.shape[1]
        height = self.trained_albedo.shape[0]

        self.loss_output = spy.Tensor.empty(app.device, (width, height), 'float3')
        self.loss_output = self.mipmap_module.calculate_grad(0, 
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
    
    def update_loss(self):        
        if self.is_training == False:
            return
        #loss calculation using np
        loss_np = self.loss_output.to_numpy()
        loss_max = np.max(loss_np)
        loss = np.mean(loss_np) 

        if self.is_training is True:
            self.loss_text.text = self.loss_format.format(loss, loss_max)
            self.loss_history.add(loss, self.current_epoch)
            self.loss_history.visualize()

    def tick(self, app) :
        if app.mouse_down:
            delta = app.mouse_pos - self.prev_mouse_pos
            self.view_offset += spy.int2(int(delta.x), int(delta.y))
        self.prev_mouse_pos = app.mouse_pos

        should_run_training = self.is_training and self.current_epoch < self.max_epoch
        #create random light direction with numpy
        if should_run_training :
            light_dir_np = self.presampled_light_dirs[self.current_light_dir_index]
            self.set_light_dir(light_dir_np)
            self.ref_output = self.render_reference(app, self.light_dir)

            self.train(app, self.light_dir)

            self.current_light_dir_index += 1
            if self.current_light_dir_index >= len(self.presampled_light_dirs):
                self.current_light_dir_index = 0
                self.current_epoch += 1
                self.update_loss()
                self.progressbar.fraction = self.current_epoch / self.max_epoch
                if self.current_epoch >= self.max_epoch:
                    self.pause_train()
         
    def render(self, app: App):
        if self.ref_output is None:
            self.ref_output = self.render_reference(app, self.light_dir)
        self.mipmap_module.clear(spy.float4(0.0), app.output_texture)

        width = self.trained_albedo.shape[1]
        height = self.trained_albedo.shape[0]
        
        self.rendered_output = spy.Tensor.empty(app.device, (width, height), 'float3')
        self.rendered_output = self.mipmap_module.render(pixel=spy.call_id(),
            material = {
                "albedo": self.trained_albedo,
                "normal": self.trained_normal,
                "roughness": self.trained_roughness,
                "metallic": self.metallic
            },
            light_dir=self.light_dir,
            view_dir=spy.float3(0, 0, 1),
            _result=self.rendered_output
            )        
        out_size = spy.int2((int)(self.ref_output.shape[0] * self.view_scale), (int)(self.ref_output.shape[1] * self.view_scale))
        xpos = int(self.view_offset.x * self.view_scale)
        ypos = int(self.view_offset.y * self.view_scale)

        out_size = spy.int2((int)(out_size.x / VIEW_SIZE_SCALE), (int)(out_size.y / VIEW_SIZE_SCALE))
        if self.view_mode == 0: # Rendered->Downsampled
            # albedo,normal, roughness, rendered order
            self.blit(self.albedo_map, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
            self.blit(self.albedo_map, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
            xpos += out_size.x + 10
            self.blit(self.normal_map, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
            xpos += out_size.x + 10
            self.blit1(self.roughness_map, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
            xpos += out_size.x + 20
            self.blit(self.ref_output, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=True, bilinear=False)
        elif self.view_mode == 1: # Downsampled->Rendered
            rendered_after_downsampled = spy.Tensor.empty(app.device, (width, height), 'float3')
            self.mipmap_module.render(pixel=spy.call_id(),
                material = {
                    "albedo": self.downsampled_albedo,
                    "normal": self.downsampled_normal,
                    "roughness": self.downsampled_roughness,
                    "metallic": self.metallic
                },
                light_dir=self.light_dir,
                view_dir=spy.float3(0, 0, 1),
                _result=rendered_after_downsampled
                )

            self.blit(self.downsampled_albedo, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
            xpos += out_size.x + 10
            self.blit(self.downsampled_normal, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
            xpos += out_size.x + 10
            self.blit1(self.downsampled_roughness, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=True, bilinear=False)
            xpos += out_size.x + 20
            self.blit(rendered_after_downsampled, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=True, bilinear=False)

        xpos = int(self.view_offset.x * self.view_scale)
        ypos += out_size.y + VERTICAL_SPACING

        self.blit(self.trained_albedo, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)

        self.blit(self.trained_albedo, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
        xpos += out_size.x + 10
        self.blit(self.trained_normal, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
        xpos += out_size.x + 10
        self.blit1(self.trained_roughness, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=False, bilinear=False)
        xpos += out_size.x + 20
        if self.rendered_output is not None:
            self.blit(self.rendered_output, app.output_texture, size=out_size, offset=spy.int2(xpos, ypos), tonemap=True, bilinear=False)


        return super().render(app)

    def post_render(self, app):
        return super().post_render(app)
    


app = App()
renderer = MipmapRenderer(app)
app.set_renderer(renderer)



app.run()
