from textual.events import Enter
from textual.events import Focus
from textual.events import Event
from textual.containers import HorizontalGroup
import os
import time

import cv2
import cv2_enumerate_cameras as cv2ec
from PIL import Image as PILImage
from numpy.typing import NDArray

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Center, Middle
from textual.events import MouseScrollUp, MouseScrollDown
from textual.validation import Number
from textual.reactive import reactive
from textual.widgets import Input, Select, Footer, Header
from textual_image.widget import Image

import yaml

"""
TODO:
    [x] Camera source selector
    [x] Camera column styling
    [x] Importing config
    [x] Exporting config
"""

class Calibration(App):
    SRC_SELECT_ID = "src-select"
    ADJUSTER_ID = "adjuster"
    PREVIEW_SIZE = (640, 480)
    DEFAULT_CSS = """
    #left-container {
        margin: 1 0;
    }
    #right-container {
        margin: 1 0;
    }

    #main-frame {
        margin: 1 0;
        width: auto;
        height: 50vh;
    }

    .inline-widget {
        width: 50vw;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit calibration"),
        ("ctrl+s", "save", "Save information"),
    ]
    DEFAULT_SAVE_DIR = "."
    DEFAULT_SAVE_FILE = "config.yaml"

    threshold = reactive(0.1)
    ADJ_MOUSE_DELTA = 0.01

    @property
    def cfg_path(self):
        return os.path.join(self.DEFAULT_SAVE_DIR, self.DEFAULT_SAVE_FILE)

    def action_save(self):
        os.makedirs(self.DEFAULT_SAVE_DIR, exist_ok=True)
        config = {
            "camIndex": self.cam_index,
            "threshold": round(self.threshold, 2)
        }

        with open(self.cfg_path, "w") as f:
            yaml.dump(config, f, indent=4)

        self.notify(f"Info saved to {self.cfg_path}", title="Config")

    def on_mount(self):
        self.set_interval(1/30, self.update_frame)

    @staticmethod
    def focus_peak(cvframe, threshold, highlight: tuple[int, int, int]):
        # Grayscale
        gray = cv2.cvtColor(cvframe, cv2.COLOR_BGR2GRAY)

        # TODO(mabay): Not sure if blur is needed here
        # gauss_blur = ski.filters.gaussian(gray, sigma=0.4)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(gx, gy)

        # Threshold
        mask = edges > (threshold * edges.max())

        # Apply to original image
        hr, hg, hb = highlight
        cvframe[mask, 0] = hr
        cvframe[mask, 1] = hg
        cvframe[mask, 2] = hb
        return cvframe

    def update_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            return

        resized_image = cv2.resize(frame, self.PREVIEW_SIZE)
        # NOTE(mabay): OpenCV stores its images in BGR format, for some reason.
        processed = Calibration.focus_peak(frame, self.threshold, (0, 255, 0))
        self.frame_img.image = PILImage.fromarray(
            cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        select = event.select
        if select.id == self.SRC_SELECT_ID:
            self.cam_index = select.value
            self.cam = cv2.VideoCapture(self.cam_index)

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        if self.focused.id == self.ADJUSTER_ID:
            self.threshold += self.ADJ_MOUSE_DELTA
            self.threshold = min(1, self.threshold)
            self.query_one(f"#{self.ADJUSTER_ID}").value = f"{self.threshold:.2f}"

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        if self.focused.id == self.ADJUSTER_ID:
            self.threshold -= self.ADJ_MOUSE_DELTA
            self.threshold = max(0, self.threshold)
            self.query_one(f"#{self.ADJUSTER_ID}").value = f"{self.threshold:.2f}"


    def compose(self) -> ComposeResult:
        # Import config on startup
        # Default config
        config = {
            "camIndex": 0,
            "threshold": 0.1
        }
        if os.path.exists(self.cfg_path):
            with open(self.cfg_path, "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)

        self.threshold = config["threshold"]
        self.cam_index = config["camIndex"]

        self.cam = cv2.VideoCapture(self.cam_index)
        cams = list(cv2ec.enumerate_cameras())
        cam_options = [(f"[{cam.index}] {cam.name}", cam.index) for cam in cams]
        mask_options = [("red", 0), ("green", 1), ("blue", 2)]

        yield Header()
        # NOTE(mabay): For some reason doing this same thing using CSS seems impossible.
        with Middle():
            with HorizontalGroup():
                with Center():
                    with Container(id="left-container"):
                        yield Select(
                            cam_options,
                            prompt="Camera Source",
                            classes="inline-widget",
                            id=self.SRC_SELECT_ID,
                        )

                        self.frame_img = Image(id="main-frame")
                        yield self.frame_img

                        yield Input(
                            value=str(self.threshold),
                            placeholder="Threshold adjustment",
                            id=self.ADJUSTER_ID,
                            classes="inline-widget",
                            validators=[Number(minimum=0.0, maximum=1.0)],
                        )

                    # with Container(id="right-container"):
                    #     yield Select(mask_options, prompt="Color Masks", allow_blank=False)

        yield Footer()


if __name__ == "__main__":
    Calibration().run()