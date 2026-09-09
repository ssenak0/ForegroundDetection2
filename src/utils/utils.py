import cv2
import numpy as np
from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.logger import LoggerManager


class ModelLoader:

    class BackgroundSubtractorWrapper:
        def __init__(self, cv_model, learning_rate):
            self.model = cv_model
            self.learning_rate = learning_rate

        def apply(self, image):
            return self.model.apply(image, learningRate=self.learning_rate)

    class FrameDifferencingWrapper:
        def __init__(self):
            self.prev_gray = None

        def apply(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None:
                self.prev_gray = gray
                return np.zeros_like(gray)

            diff = cv2.absdiff(gray, self.prev_gray)
            self.prev_gray = gray
            return diff

    class RunningAverageWrapper:
        def __init__(self, alpha=0.05,  bg_type="mean"):
            self.alpha = alpha
            self.bg = None
            self.bg_type = bg_type
            self.frame_buffer = []
            self.frame_count = 0
            self.init_frames = 0

        def apply(self, image, init_frames):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self.frame_count < init_frames:
                self.frame_buffer.append(gray.astype(np.float32))
                self.frame_count += 1
                return np.zeros_like(gray)

            if self.bg is None:
                stack = np.stack(self.frame_buffer)
                if self.bg_type == "median":
                    self.bg = np.median(stack, axis=0).astype(np.float32)
                else:
                    self.bg = np.mean(stack, axis=0).astype(np.float32)
                self.frame_buffer = None

            cv2.accumulateWeighted(gray, self.bg, self.alpha)
            bg_uint8 = cv2.convertScaleAbs(self.bg)
            diff = cv2.absdiff(gray, bg_uint8)
            return diff

    def __init__(self, config: dict):
        self.config = config
        self.application = Application()
        self.logger = LoggerManager()

    def load_model(self):
        model_type = self.application.get_param(self.config, "type")

        if model_type == "MOG2":
            model = cv2.createBackgroundSubtractorMOG2(
                history=self.application.get_param(self.config, "history"),
                varThreshold=self.application.get_param(self.config, "varThreshold"),
                detectShadows=self.application.get_param(self.config, "detectShadows")
            )
            return self.BackgroundSubtractorWrapper(
                model,
                self.application.get_param(self.config, "learningRate")
            )

        if model_type == "KNN":
            model = cv2.createBackgroundSubtractorKNN(
                history=self.application.get_param(self.config, "history"),
                dist2Threshold=self.application.get_param(self.config, "dist2Threshold"),
                detectShadows=self.application.get_param(self.config, "detectShadows")
            )
            return self.BackgroundSubtractorWrapper(
                model,
                self.application.get_param(self.config, "learningRate")
            )

        if model_type == "FrameDifferencing":
            return self.FrameDifferencingWrapper()

        if model_type == "RunningAverage":
            return self.RunningAverageWrapper(
                self.application.get_param(self.config, "learningRate"),
                self.application.get_param(self.config, "bgType")
            )

        raise ValueError(f"Unsupported model type: {model_type}")
