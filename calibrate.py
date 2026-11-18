import os
import math
import operator
from typing import Any, Callable

import cv2
import cv2_enumerate_cameras as cv2ec
from cv2_enumerate_cameras.camera_info import CameraInfo
from PIL import Image as PILImage
import numpy as np
from numpy.typing import NDArray

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, HorizontalGroup
from textual.events import MouseScrollUp, MouseScrollDown
from textual.validation import Number
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, Static, Select, Footer, Header, TabPane, TabbedContent
from textual_image.widget import Image

import yaml

"""
TODO:
    [x] Camera source selector
    [x] Camera column styling
    [x] Importing config
    [x] Exporting config

    [x] Resizing to smaller -> focus peaking -> resize to larger -> display
    [x] Make camera focus handle its own import and export
    [x] Make both tabs share a single camera context
    [x] Release camera on unmount

    [ ] Extract scrollable input to its own widget
        - Makes it easy to re-use in the camera mask threshold range
    [ ] Create a red and green mask (these will be the builtin ones)
    [ ] Add way to add more masks easily
        - Look at stopwatch example
"""

DEFAULT_SAVE_DIR = "."


class ScrollableInput(Input):
    def __init__(self, value: float = 0.0, range_min: float = 0.0, range_max: float = 1.0,
        placeholder="num", type="number", id=None,
        validators=[Number(minimum=0.0, maximum=1.0)], **kwargs):
        super().__init__(
            value=f"{value:.2f}",
            placeholder=placeholder,
            type=type,
            id=id,
            validators=validators,
        )

        self.range_min = min(range_min, range_max)
        self.range_max = max(range_min, range_max)
        self.delta = 0.01 * (self.range_max - self.range_min)

    def on_mouse_scroll_up(self):
        new_val = min(self.range_max, float(self.value) + self.delta)
        self.value = f"{new_val:.2f}"

    def on_mouse_scroll_down(self):
        new_val = max(self.range_min, float(self.value) - self.delta)
        self.value = f"{new_val:.2f}"


class CameraFocus(Widget):
    SRC_SELECT_ID = "src-select"
    ADJUSTER_ID = "adjuster"

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
        self.config = {
            "camIndex": 0,
            "threshold": 0.15
        }

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

    def compose(self) -> ComposeResult:
        available_cams = cv2ec.enumerate_cameras()
        cam_options = [(f"[{cam.index}] {cam.name}", cam.index)
            for cam in available_cams]
        cam_indices = [c[1] for c in cam_options]

        if os.path.exists(self.cfg_path):
            with open(self.cfg_path, "r") as f:
                self.config = yaml.load(f, Loader=yaml.Loader)

        self.cam_index = self.config.get("camIndex", 0)
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
                ScrollableInput(
                    value=self.config.get("threshold", 0.15),
                    placeholder="min",
                    id=self.ADJUSTER_ID,
                ),
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
            self.cam_index = int(select.value) if select.value else 0
            self.update_cam(self.cam_index)

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        self.threshold = float(event.input.value)

    def watch_threshold(self, threshold) -> None:
        if not self.is_mounted:
            return
        self.query_one(f"#{self.ADJUSTER_ID}").value = f"{threshold:.2f}"


class MaskConfig(Widget):
    MIN_ADJUSTER = "min-adjuster"
    MAX_ADJUSTER = "max-adjuster"
    DEFAULT_CSS = """
    MaskConfig {
        width: 50%;
        height: auto;
        align: center middle;
        Label {
            background: red;
            width: auto;
        }
        Input {
            width: 1fr;
        }
    }
    """

    def compose(self):
        # TODO(mabay): Implement an Input validator that ensures $min < max$

        yield Horizontal(
            Label("Red"),
            ScrollableInput(
                value=0.0,
                placeholder="min_l",
            ),
            ScrollableInput(
                value=0.0,
                placeholder="min_a",
            ),
            ScrollableInput(
                value=0.0,
                placeholder="min_b",
            ),
        )




class CameraMask(Widget):
    PREVIEW_HEIGHT = 150
    PREVIEW_ASPECT_RATIO = 4/3

    DEFAULT_CSS = """
    #container {
        width: 50%;
        align: center middle;
        padding: 1;
        #orig-frame, #masked-frame {
            width: 1fr;
            height: auto;
            margin: 1;
        }
    }
    """

    def __init__(self, update_cam: Callable[[int], None], **kwargs):
        super().__init__(**kwargs)
        self.update_cam = update_cam
        self.orig_img = Image(id="orig-frame")
        self.masked_img = Image(id="masked-frame")
        self.lab_data: dict[str, Any] = {
            "red": {
                # "min": (0, 128, 128),
                # "max": (255, 128, 128),
                "min": (40, 160, 130),
                "max": (255, 255, 205),
            }
        }

    def compose(self):
        yield Container(
            HorizontalGroup(
                self.orig_img,
                self.masked_img,
            ),
            MaskConfig(),
            id="container"
        )

    def frame_update(self, frame):
        fh, _ = frame.shape[:2]
        preview_size = (int(self.PREVIEW_ASPECT_RATIO*fh), fh)
        resized_frame = cv2.resize(frame, preview_size,
            interpolation=cv2.INTER_NEAREST)
        resized_clean = resized_frame.copy()
        annotated_image = resized_frame.copy()

        resized_clean = cv2.GaussianBlur(resized_clean, (3, 3), 3)
        # Convert to LAB space
        resized_clean = cv2.cvtColor(resized_clean, cv2.COLOR_BGR2LAB)

        # If we're calling target_contours() multiple times, some args will
        # be the same. Let's put them here to re-use them.
        contour_args = {
            'open_kernel': np.ones((3, 3), np.uint8),
            'close_kernel': np.ones((3, 3), np.uint8),
        }
        # extract the LAB threshold
        target_color = "red"
        masks = {}

        threshold = (tuple(self.lab_data[target_color][key]) for key in ['min', 'max'])
        masks.setdefault(target_color, {})
        # run contour detection
        target_contours = self.color_contour_detection(
            resized_clean,
            tuple(threshold),  # type: ignore
            save_to_dict=masks[target_color],
            **contour_args
        )
        biggest_contour, biggest_contour_area = target_contours[0] if target_contours else (None, 0)
        # did we detect something of interest?
        detected = biggest_contour_area > 300

        # draw annotations of detected contours
        target_color_color = (255, 0, 0)
        black_color = (0, 0, 0)

        if detected:
            self.draw_fitted_rect(annotated_image, biggest_contour, target_color_color)
            self.draw_text(annotated_image, target_color_color, target_color)
        else:
            self.draw_text(annotated_image, black_color, 'None')


        self.orig_img.image = PILImage.fromarray(
            cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        self.masked_img.image = PILImage.fromarray(
            cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def color_contour_detection(
        frame,
        threshold: tuple[tuple[int, int, int], tuple[int, int, int]],
        open_kernel: NDArray | None = None,
        close_kernel: NDArray | None = None,
        save_to_dict=None,
    ):
        if save_to_dict is None:
            save_to_dict = {}
        # Image Processing
        # mask the colors we want
        threshold = [tuple(li) for li in threshold]  # cast to tuple to make cv2 happy
        frame_mask = cv2.inRange(frame, *threshold)  # type: ignore
        save_to_dict['mask'] = frame_mask
        # Perform an opening and closing operation on the mask
        # https://youtu.be/1owu136z1zI?feature=shared&t=34
        frame = frame_mask.copy()
        if open_kernel is not None:
            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, open_kernel)
        if close_kernel is not None:
            frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # find contours (blobs) in the mask
        contours = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        save_to_dict['contours'] = contours
        areas = [math.fabs(cv2.contourArea(contour)) for contour in contours]
        # zip to provide pairs of (contour, area)
        zipped = zip(contours, areas)
        # return largest-to-smallest contour
        return sorted(zipped, key=operator.itemgetter(1), reverse=True)

    @staticmethod
    def draw_fitted_rect(img, contour, color):
        # draw rotated fitted rectangle around contour
        rect = cv2.minAreaRect(contour)
        box = np.int_(cv2.boxPoints(rect))
        cv2.drawContours(img, [box], -1, color, 2)

    @staticmethod
    def draw_text(img, color, name):
        # Print the detected color on the screen
        cv2.putText(img, f"Color: {name}", (10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


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

    def on_unmount(self):
        self.cam.release()

    def tick(self):
        ret, frame = self.cam.read()
        if not ret:
            assert False, (ret, frame.shape)
            return

        # TODO(mabay): Find a better way to ensure that only one of these updates
        # function is triggered.
        match self.query_one("#tabs-container").active:
            case "focus-tab": self.camera_focus.frame_update(frame)
            case "mask-tab" : self.camera_mask.frame_update(frame)

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


if __name__ == "__main__":
    Calibration().run()