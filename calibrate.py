import os
import time
from typing import Callable

import cv2
import cv2_enumerate_cameras as cv2ec
from cv2_enumerate_cameras.camera_info import CameraInfo
from PIL import Image as PILImage
from numpy.typing import NDArray

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, HorizontalGroup
from textual.events import MouseScrollUp, MouseScrollDown
from textual.validation import Number
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Select, Footer, Header, TabPane, TabbedContent
from textual_image.widget import Image

import yaml

"""
TODO:
    [x] Camera source selector
    [x] Camera column styling
    [x] Importing config
    [x] Exporting config

    [x] Resizing to smaller -> focus peaking -> resize to larger -> display
    [ ] Make camera focus handle its own import and export
    [ ] Make both tabs share a single camera context
    [ ] Release camera on unmount
"""

DEFAULT_SAVE_DIR = "."


class CameraFocus(Widget):
    SRC_SELECT_ID = "src-select"
    ADJUSTER_ID = "adjuster"

    threshold = reactive(0.1)
    HIGHLIGHT_COLOR = (0, 255, 0)
    ADJ_MOUSE_DELTA = 0.01

    PREVIEW_HEIGHT = 200
    PREVIEW_ASPECT_RATIO = 4/3
    DEFAULT_SAVE_FILE = "focus_config.yaml"

    DEFAULT_CSS = """
    #container {
        width: 50%;
        padding: 1;
        #main-frame {
            width: 100%;
            height: auto;
            margin: 0 0 1 0;
        }
        Select {
            width: 1fr;
        }
        Input {
            width: auto;
        }
    }
    """

    def __init__(self, update_cam: Callable[[int], None], **kwargs):
        super().__init__(**kwargs)
        self.update_cam = update_cam
        self.frame_img = Image(id="main-frame")

    @property
    def cfg_path(self):
        return f"{DEFAULT_SAVE_DIR}/{self.DEFAULT_SAVE_FILE}"

    def export_config(self, save_dir: str):
        # TODO(mabay): Add export of camera focus config
        os.makedirs(save_dir, exist_ok=True)
        config = {
            "camIndex": self.cam_index,
            "threshold": round(self.threshold, 2)
        }

        with open(self.cfg_path, "w") as f:
            yaml.dump(config, f, indent=4)

        self.notify(f"Info saved to {self.cfg_path}", title="Config")

    def import_config(self):
        # Default config
        config = {}
        if os.path.exists(self.cfg_path):
            with open(self.cfg_path, "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)

        self.cam_index = config.get("camIndex", 0)
        self.threshold = config.get("threshold", 0.15)

    def compose(self) -> ComposeResult:
        available_cams = cv2ec.enumerate_cameras()
        cam_options = [(f"[{cam.index}] {cam.name}", cam.index)
            for cam in available_cams]
        cam_indices = [c[1] for c in cam_options]

        self.import_config()
        if self.cam_index == 0:
            self.cam_index = cam_indices[0]
        elif self.cam_index not in cam_indices:
            raise ValueError(f"Unknown camera index selected: {self.cam_index}")

        self.update_cam(self.cam_index)

        yield Container(
            self.frame_img,
            HorizontalGroup(
                Select(
                    cam_options,
                    value=self.cam_index,
                    prompt="Camera Source",
                    id=self.SRC_SELECT_ID,
                ),
                Input(
                    value=f"{self.threshold:.2f}",
                    placeholder="Threshold adjustment",
                    type="number",
                    id=self.ADJUSTER_ID,
                    validators=[Number(minimum=0.0, maximum=1.0)],
                )
            ),
            id="container"
        )

    def frame_update(self, frame):
        """ Apply focus peaking algorithm """
        # Downscale original frame
        fh, fw = frame.shape[:2]
        small_frame = cv2.resize(frame,
            (int(self.PREVIEW_ASPECT_RATIO*self.PREVIEW_HEIGHT), self.PREVIEW_HEIGHT),
            interpolation=cv2.INTER_NEAREST)

        # Grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # TODO(mabay): Not sure if blur is needed here
        # gauss_blur = ski.filters.gaussian(gray, sigma=0.4)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(gx, gy)

        # Threshold
        mask = edges > (self.threshold * edges.max())

        # Apply to highlight to edges on the small frame
        hr, hg, hb = self.HIGHLIGHT_COLOR
        small_frame[mask, 0] = hr
        small_frame[mask, 1] = hg
        small_frame[mask, 2] = hb

        # Upscale frame back to the original size
        resized_out = cv2.resize(small_frame, (int(self.PREVIEW_ASPECT_RATIO*fh), fh),
            interpolation=cv2.INTER_NEAREST)
        self.frame_img.image = PILImage.fromarray(
            cv2.cvtColor(resized_out, cv2.COLOR_BGR2RGB))

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        select = event.select
        if select.id == self.SRC_SELECT_ID:
            self.cam_index = select.value
            self.update_cam(self.cam_index)

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        self.threshold = float(event.input.value)

    def inc_threshold(self, focused: Widget) -> None:
        if focused.id == self.ADJUSTER_ID:
            self.threshold += self.ADJ_MOUSE_DELTA
            # Clamp to a maximum of 1 (aka. 100%)
            self.threshold = min(1, self.threshold)
            self.query_one(f"#{self.ADJUSTER_ID}").value = f"{self.threshold:.2f}"

    def dec_threshold(self, focused: Widget) -> None:
        if focused.id == self.ADJUSTER_ID:
            self.threshold -= self.ADJ_MOUSE_DELTA
            # Clamp to a minimum of 0 (aka. 0%)
            self.threshold = max(0, self.threshold)
            self.query_one(f"#{self.ADJUSTER_ID}").value = f"{self.threshold:.2f}"


class CameraMask(Widget):
    PREVIEW_HEIGHT = 200
    PREVIEW_ASPECT_RATIO = 4/3

    DEFAULT_CSS = """
    #container {
        width: 80%;
        align: center middle;
        padding: 1;
        #orig-frame , #masked-frame {
            width: 1fr;
            height: auto;
            margin: 1;
        }
    }
    """

    def __init__(self, update_cam: Callable[[int], None], **kwargs):
        super().__init__(**kwargs)
        self.update_cam = update_cam

    def compose(self):
        self.orig_img = Image(id="orig-frame")
        self.masked_img = Image(id="masked-frame")

        yield Container(
            Horizontal(
                self.orig_img,
                self.masked_img,
            ),
            id="container"
        )

    def frame_update(self, frame):
        fh, _ = frame.shape[:2]
        resized_frame = cv2.resize(frame, (int(self.PREVIEW_ASPECT_RATIO*fh), fh),
            interpolation=cv2.INTER_NEAREST)
        self.orig_img.image = PILImage.fromarray(
            cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        self.masked_img.image = PILImage.fromarray(
            cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))


class Calibration(App):
    DEFAULT_CSS = """
    TabbedContent { height: 1fr; }
    TabPane { height: 1fr; }
    CameraFocus, CameraMask { width: 1fr; height: 1fr; }
    """

    BINDINGS = [
        ("ctrl+s", "save_config", "Save config"),
    ]

    def action_save_config(self):
        self.camera_focus.export_config(DEFAULT_SAVE_DIR)

    def on_mount(self):
        self.set_interval(1/30, self.tick)

    def tick(self):
        ret, frame = self.cam.read()
        if not ret:
            return

        match self.query_one("#tabs-container").active:
            case "focus-tab":
                self.camera_focus.frame_update(frame)
            case "mask-tab":
                self.camera_mask.frame_update(frame)

    def update_camera(self, cam_index: int) -> None:
        self.cam.release()
        self.cam = cv2.VideoCapture(cam_index)

    def compose(self) -> ComposeResult:
        self.cam = cv2.VideoCapture(0)
        self.camera_focus = CameraFocus(self.update_camera)
        self.camera_mask = CameraMask(self.update_camera)

        yield Header()
        with TabbedContent(id="tabs-container"):
            with TabPane("Focus", id="focus-tab"):
                yield self.camera_focus
            with TabPane("Masks", id="mask-tab"):
                yield self.camera_mask
        yield Footer()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        if self.focused:
            self.camera_focus.inc_threshold(self.focused)

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        if self.focused:
            self.camera_focus.dec_threshold(self.focused)

if __name__ == "__main__":
    Calibration().run()