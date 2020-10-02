from .Detector import Detector
from core import Pipeline
from core.common import get_diff, draw_contours,find_contours
from distances import max_distance
import numpy as np
import cv2

from functools import partial


class BackgroundNoGaussian(Detector):

    # apply preprocessing manually
    def get_preprocessing(self):
        return None

    def blurframe(self, frame):
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def get_pipeline(self, debug=False):
        bg = self.video.get_background(np.median, 100)

        p = Pipeline(debug)

        p.add_operation("Input", lambda frame: frame, hide=True)

        def split(frame):
            return frame, self.blurframe(frame.copy())

        gettext = lambda t: [f"{t}_NO_BLUR", f"{t}_BLUR"]

        p.add_operation(gettext("INPUT"), split)

        def compute_diff(frames):
            no_blur, blur = frames
            diff1 = get_diff([bg, no_blur], max_distance, lambda diff: diff > 30)
            diff2 = get_diff([self.blurframe(bg), blur], max_distance, lambda diff: diff > 30)
            return diff1, diff2

        p.add_operation(gettext("DIFF"), compute_diff)

        # # remove noise
        # def remove_noise(frames):
        #     no_blur, blur = frames
        #     img1 = cv2.morphologyEx(no_blur, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        #                             iterations=1)
        #     img2 = cv2.morphologyEx(blur, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        #                             iterations=1)
        #     return img1, img2
        #
        # p.add_operation(gettext("Remove noise"), remove_noise)
        #
        # # connect vertically
        # def connect_vertically(frames):
        #     no_blur, blur = frames
        #     img1 = cv2.morphologyEx(no_blur, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        #                             iterations=1)
        #     img2 = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        #                             iterations=1)
        #     return img1, img2
        #
        # p.add_operation(gettext("Connect figures vertically"), connect_vertically)
        #
        # # fill holes
        # def fill_holes(frames):
        #     no_blur, blur = frames
        #     img1 = cv2.morphologyEx(no_blur, cv2.MORPH_CLOSE,
        #                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
        #                                                       (10, 10)),
        #                             iterations=5)
        #     img2 = cv2.morphologyEx(blur, cv2.MORPH_CLOSE,
        #                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
        #                                                       (10, 10)),
        #                             iterations=5)
        #     return img1, img2
        #
        # p.add_operation(gettext("Fill holes"), fill_holes)
        #
        # # reduce boundaries
        # def reduce_boundaries(frames):
        #     no_blur, blur = frames
        #     img1 = cv2.erode(no_blur, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        #     img2 = cv2.erode(blur, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        #     return img1, img2
        #
        # p.add_operation(gettext("Reduce boundaries"), reduce_boundaries)

        def draw_cnts(frames):
            no_blur, blur = frames
            img1 = draw_contours(find_contours(no_blur), p.input)
            img2 = draw_contours(find_contours(blur), p.input)
            return img1, img2

        p.add_operation(gettext("OUTPUT"), draw_cnts)

        return p
