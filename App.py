# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

class Renderer:
    def tick(self, app) -> None : ...
    def pre_render(self, app) -> None : ...
    def render(self, app) -> None : ...
    def post_render(self, app) -> None : ...

class App:
    def __init__(self):
        super().__init__()
        self.window = spy.Window(width=1920, height=1024, title="Example", resizable=True)

        device_type = spy.DeviceType.automatic
        self.device = spy.create_device(
            device_type, enable_debug_layers=False, include_paths=[EXAMPLE_DIR]
        )
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        self.ui = spy.ui.Context(self.device)

        self.output_texture = None

        self.mouse_pos = spy.float2()
        self.mouse_down = False

        self.playing = True
        self.fps_avg = 0.0
        self.tps_avg = 0.0

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.renderer = None

        self.max_fps = 60

        self.setup_ui()

    def set_renderer(self, renderer:Renderer):
        self.renderer = renderer

    def setup_ui(self):
        screen = self.ui.screen
        self.ui_window = spy.ui.Window(screen, "Settings", size=spy.float2(400, 540), position=spy.float2(1400, 5))

        self.tick_rate_text = spy.ui.Text(self.ui_window, "TPS:")
        self.fps_text = spy.ui.Text(self.ui_window, "FPS: 0")

        self.max_fps_slider = spy.ui.SliderInt(self.ui_window, "Max FPS", value=self.max_fps, callback=self.on_max_fps_changed, min=15, max=60)
        
    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return

        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
            elif event.key == spy.KeyCode.f1:
                if self.output_texture:
                    spy.tev.show_async(self.output_texture)
            elif event.key == spy.KeyCode.f2:
                if self.output_texture:
                    bitmap = self.output_texture.to_bitmap()
                    bitmap.convert(
                        spy.Bitmap.PixelFormat.rgb,
                        spy.Bitmap.ComponentType.uint8,
                        srgb_gamma=True,
                    ).write_async("screenshot.png")

    def on_mouse_event(self, event: spy.MouseEvent):
        if self.ui.handle_mouse_event(event):
            return

        if event.type == spy.MouseEventType.move:
            self.mouse_pos = event.pos
        elif event.type == spy.MouseEventType.button_down:
            if event.button == spy.MouseButton.left:
                self.mouse_down = True
                self.drag_start = self.mouse_pos
        elif event.type == spy.MouseEventType.button_up:
            if event.button == spy.MouseButton.left:
                self.mouse_down = False

    def on_resize(self, width: int, height: int):
        self.device.wait()
        
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height, vsync=False)
        else:
            self.surface.unconfigure();

    def on_max_fps_changed(self, value: int):
        self.max_fps = value

    def run(self):
        frame = 0
        render_timer = spy.Timer()
        tick_timer = spy.Timer()


        while not self.window.should_close():            
            self.window.process_events()
            self.ui.process_events()

            delta_time = render_timer.elapsed_s()

            if self.renderer is not None:
                tick_time = tick_timer.elapsed_s()
                if tick_time >= 1.0 / 1000.0: # 1000 TPS max
                    tick_timer.reset()
                    self.tps_avg = 0.9 * self.tps_avg + 0.1 * (1.0 / tick_time if tick_time > 0 else 0.0)
                    self.tick_rate_text.text = f"TPS: {self.tps_avg:.1f}"
            
                    self.renderer.tick(self)

            if self.surface.config is None:
                continue

            target_frame_time = 1.0 / self.max_fps
            
            if delta_time < target_frame_time:
                continue

            render_timer.reset()
            # FPS counter
            current_fps = 1.0 / delta_time
            self.fps_avg = 0.9 * self.fps_avg + 0.1 * current_fps

            self.fps_text.text = f"FPS: {self.fps_avg:.1f}"

            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            if (
                self.output_texture == None
                or self.output_texture.width != surface_texture.width
                or self.output_texture.height != surface_texture.height
            ):
                self.output_texture = self.device.create_texture(
                    format=spy.Format.rgba16_float,
                    width=surface_texture.width,
                    height=surface_texture.height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                    label="output_texture",
                )

            if self.renderer is not None:
                self.renderer.pre_render(self)
                self.renderer.render(self)
                self.renderer.post_render(self)

            command_encoder = self.device.create_command_encoder()
            command_encoder.blit(surface_texture, self.output_texture)

            self.ui.new_frame(surface_texture.width, surface_texture.height)
            self.ui.render(surface_texture, command_encoder)

            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture

            self.surface.present()

            frame += 1