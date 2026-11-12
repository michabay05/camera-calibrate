from numpy.typing import NDArray
import numpy as np
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Label
from textual_image.widget import Image

from PIL import Image as PILImage
import skimage as ski
import cv2


class CameraOutput(Widget):
    DEFAULT_CSS = """
    CameraOutput {
        border: round gray;
        width: 1fr;
        height: 1fr;
        margin: 1;
        align: center middle;

        Image {
            width: auto;
            height: auto;
        }
    }
    """
    def compose(self) -> ComposeResult:
        self.output_image = Image()
        yield self.output_image

    def show_frame(self, frame) -> None:
        self.output_image.image = PILImage.fromarray(frame)


class CameraProcessed(Widget):
    DEFAULT_CSS = """
    CameraProcessed {
        border: round gray;
        width: 1fr;
        height: 1fr;
        margin: 1;
        align: center middle;

        Image {
            width: auto;
            height: auto;
        }
    }
    """

    last_frame: NDArray | None = None
    threshold = reactive(0.2)
    highlight = (0, 255, 0)

    def compose(self) -> ComposeResult:
        self.processed_image = Image()
        yield self.processed_image

    def show_frame(self, frame: NDArray) -> None:
        self.last_frame = frame.copy()
        self.focus_peak()

    def watch_threshold(self) -> None:
        self.focus_peak()

    def focus_peak(self):
        if self.last_frame is None:
            return None

        # Grayscale
        gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)

        # TODO(mabay): Not sure if blur is needed here
        # gauss_blur = ski.filters.gaussian(gray, sigma=0.4)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(gx, gy)

        # Threshold
        mask = edges > (self.threshold * edges.max())

        # Apply to original image
        proc_img = self.last_frame
        hr, hg, hb = self.highlight
        proc_img[mask, 0] = hr
        proc_img[mask, 1] = hg
        proc_img[mask, 2] = hb
        self.processed_image.image = PILImage.fromarray(proc_img)


class Adjusters(HorizontalGroup):
    DEFAULT_CSS = """
    Adjusters {
        height: 3;
        width: 1fr;
        align: center middle;

        Label {
            height: 3;
            width: auto;
            padding: 0 1;
            content-align: center middle;
        }

        Label.name {
            text-style: bold;
        }

        Label.value {
            width: 6;
            padding: 0;
            margin: 0 3;
            content-align: right middle;
        }

        Button {
            width: 5;
            min-width: 5;
            padding: 0 1;
        }
    }
    """

    value = reactive(0.0)
    step = 0.1
    on_change = None

    def __init__(
        self,
        name: str,
        value: float = 0.0,
        step: float = 0.1,
        on_change=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label = name
        self.value = value
        self.step = step
        self.on_change = on_change

    def compose(self) -> ComposeResult:
        yield Label(self.label, classes="name")
        yield Button("-", classes="dec")
        yield Label(f"{self.value:.1f}", classes="value")
        yield Button("+", classes="inc")

    def on_mount(self) -> None:
        try:
            label = self.query_one("Label.value", Label)
            label.update(f"{self.value:.1f}")
        except Exception:
            pass

    def watch_value(self, value: float) -> None:
        try:
            label = self.query_one("Label.value", Label)
            label.update(f"{value:.1f}")
        except Exception:
            pass
        on_change = getattr(self, "on_change", None)
        if on_change is not None:
            on_change(value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("dec"):
            self.value -= self.step
        elif event.button.has_class("inc"):
            self.value += self.step


class CameraCalibration(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cam = cv2.VideoCapture(4)

    def on_mount(self) -> None:
        # NOTE(mabay): Set to 60fps.
        self.set_interval(1 / 60, self.tick)

    def get_frame(self) -> NDArray:
        ret, frame = self.cam.read()
        if ret:
            # NOTE(mabay): OpenCV stores its images in BGR format, for some reason.
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None

    def tick(self) -> None:
        frame = self.get_frame()
        if frame is None:
            # Skip this frame for now.
            return

        new_size = (frame.shape[1]//2, frame.shape[0]//2)
        resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        self.cam_orig.show_frame(resized_frame)
        self.cam_processed.show_frame(resized_frame)

    def compose(self) -> ComposeResult:
        self.cam_orig = CameraOutput(id="orig_output")
        self.cam_processed = CameraProcessed(id="proc_out")

        yield Header()
        yield HorizontalGroup(self.cam_orig, self.cam_processed, id="cam_imgs")
        yield Footer()



if __name__ == "__main__":
    CameraCalibration().run()
