from .Detector import Detector
from core.Pipeline import Pipeline
from distances import max_distance, l1, l2
import numpy as np
import cv2

# for creating function with some arguments fixed
from functools import partial

# general operation for computing difference for any amount of frames
from core.common import get_diff
# contours related
from core.common import find_contours, draw_contours, filter_contours, get_grad, check_args

from core import Contour


class BackgroundSubtractor(Detector):
    @check_args({"kind": ["static", "first", "adaptive"]})
    def __init__(self, video, kind):
        super(BackgroundSubtractor, self).__init__(video)
        """
        Detector which use background for finding an intrusion
        :param kind: static,first,adaptive
            - static: use a static background
            - first: compute an interpolated background using first frames
            - adaptive: compute an interpolated background using previously frame
        """

        self.kind = kind

        # threshold used for computing false positive
        self.std_threshold = 1e-1

        # init parameters
        if self.kind == "static":
            self.distance = max_distance
            self.threshold = 30
        elif self.kind == "first":
            self.distance = max_distance
            self.threshold = 34
        else:
            self.distance = l1
            self.threshold = 5
            self.alpha = 0.7

        # init contours logger
        filename = f"{self.__class__.__name__}_{self.kind}.log"
        self.logger = open(filename, 'w')
        print("index",*Contour.get_info_names(), sep='\t', end='\n', file=self.logger)

    def __del__(self):
        if self.logger is not None and not self.logger.closed:
            # write to file
            self.logger.flush()
            # close handler
            self.logger.close()
            self.logger = None

    def get_preprocessing(self):
        def blur(frame):
            return cv2.GaussianBlur(frame, (5, 5), 0)

        return blur

    def _pipeline_static_first(self, debug=False):
        pipeline = Pipeline(debug=debug)

        if self.kind == "static":
            # Get static background from the video
            self.video.current_frame_position = 254
            bg = self.video.get_frame(preprocessing=self.get_preprocessing())
            self.video.current_frame_position = 0
            pipeline.store("bg", bg.copy())
            del bg
        elif self.kind == "first":
            # Get the background interpolating first 100 frames
            bg = self.video.get_background(np.median, 100, preprocessing=self.get_preprocessing(), start=0)
            pipeline.store("bg", bg.copy())
            del bg
        else:
            pass

        pipeline.add_operation("Input", lambda frame: self.get_preprocessing()(frame))

        def get_background(frame):
            bg = pipeline.load("bg")
            return bg, frame

        pipeline.add_operation(["Background", "Current frame"],
                               get_background, hide=False)

        pipeline.add_operation("Difference",
                               lambda t: get_diff(list(t), self.distance,
                                                  lambda diff: diff > self.threshold))

        def remove_noise(frame, size):
            median = cv2.medianBlur(frame, size)
            open_ = cv2.morphologyEx(median, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                     iterations=1)

            return median, open_

        pipeline.add_operation(["Median filter", "Median filter + Opening"], partial(remove_noise, size=7))

        pipeline.add_operation("Connect foreground horizontally",
                               lambda frame: cv2.morphologyEx(frame[-1], cv2.MORPH_CLOSE,
                                                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30)),
                                                              iterations=1))

        pipeline.add_operation("Connect foreground vertically",
                               lambda frame: cv2.morphologyEx(frame, cv2.MORPH_CLOSE,
                                                              cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5)),
                                                              iterations=1))

        pipeline.add_operation("Dilate for filling void",
                               lambda frame: cv2.morphologyEx(frame, cv2.MORPH_DILATE,
                                                              cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
                                                              iterations=1))
        pipeline.add_operation("Close for filling holes",
                               lambda frame: cv2.morphologyEx(frame, cv2.MORPH_CLOSE,
                                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)),
                                                              iterations=1))

        pipeline.add_operation("Output", lambda frame: self.check_contours(pipeline.input, frame, debug))

        return pipeline

    def _pipeline_adaptive(self, debug=False):
        pipeline = Pipeline(debug)

        def get_background(frame):
            bg = pipeline.load("bg")
            if bg is None: bg = frame
            bg = bg.astype(float)
            fg = frame.astype(float)
            bg = cv2.accumulateWeighted(fg, bg, self.alpha)
            pipeline.store("bg", bg)
            # return background and the current frame which are than used by get_diff([bg,fg])
            return bg.astype(np.uint8), frame

        pipeline.add_operation("Input", lambda frame: self.get_preprocessing()(frame))

        pipeline.add_operation(["Background", "Current frame"],
                               get_background, hide=False)

        pipeline.add_operation("Difference",
                               lambda t: get_diff(list(t), self.distance,
                                                  lambda diff: diff > self.threshold))

        def remove_noise(frame, median_filter_size):
            median = cv2.medianBlur(frame, median_filter_size)
            open_ = cv2.morphologyEx(median, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                                     iterations=1)

            # return median, open_
            return median, open_

        pipeline.add_operation(["Median filter", "Median filter + Opening"],
                               partial(remove_noise, median_filter_size=3))

        pipeline.add_operation("Dilate", lambda frame: cv2.morphologyEx(frame[1], cv2.MORPH_DILATE,
                                                                        cv2.getStructuringElement(cv2.MORPH_CROSS,
                                                                                                  (25, 25)),
                                                                        iterations=1))

        pipeline.add_operation("Closing", lambda frame: cv2.morphologyEx(frame, cv2.MORPH_CLOSE,
                                                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                                   (5, 5)),
                                                                         iterations=1))
        pipeline.add_operation("Erode", lambda frame: cv2.morphologyEx(frame, cv2.MORPH_ERODE,
                                                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                                 (3, 3)), iterations=1))

        pipeline.add_operation("Closing", lambda frame: cv2.morphologyEx(frame, cv2.MORPH_CLOSE,
                                                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                                   (20, 20)),
                                                                         iterations=1))

        # pipeline.add_operation("Output", lambda frame: frame.astype(np.uint8) * 255)

        pipeline.add_operation("Output", lambda mask: self.check_contours(pipeline.input, mask, debug))

        return pipeline

    def check_contours(self, input_frame, mask, debug=False):
        """
        A method that find contours, filter them and them print them on the input image
        :param input_frame: input image
        :param mask: binary image
        :return: input image with contours
        """
        # Prepare output image
        output_image = cv2.cvtColor(input_frame.copy(), cv2.COLOR_GRAY2RGB)

        # Get all parent contours filtered
        contours_to_draw = find_contours(mask, filter_contours)

        #  Get images for reporting border improvement using Canny
        if debug:
            for idx, cnt in enumerate(contours_to_draw):
                cnt_image = cnt.get_image(input_frame)
                img_canny0 = cv2.Canny(cnt_image, 40, 100, L2gradient=True)

                cnts, _ = cv2.findContours(img_canny0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if len(cnts) > 0:
                    out = cnt_image.copy()
                    out = cv2.drawContours(out, cnts, -1, (255, 0, 0))

                    import matplotlib.pyplot as plt
                    plt.imshow(np.hstack([img_canny0, out]),cmap="gray")
                    plt.show()

        # Write LOG
        # write position and amount of contours
        print(f"{self.video.current_frame_position}\t{len(contours_to_draw)}", file=self.logger if not debug else None)

        # write body
        for idx, contour in enumerate(contours_to_draw):
            contour.get_label(input_frame, self.std_threshold)
            print(contour.get_log(idx), file=self.logger if not debug else None)

            # Get images for reporting different way to compute gradient
            if debug:
                a = contour.get_image(input_frame)
                b = get_grad(a)
                local_grad_image1 = np.hstack([a, b])
                del a
                del b

                a = contour.get_image(input_frame, use_bounding_box=True)
                b = get_grad(a)
                local_grad_image2 = np.hstack([a, b])
                del a
                del b

                plt.imshow(np.vstack([local_grad_image1, local_grad_image2]), cmap="gray")
                plt.show()

        # DRAW
        output_image = draw_contours(contours_to_draw, output_image)

        return output_image

    def get_pipeline(self, debug=False):
        if self.kind in ["static", "first"]:
            return self._pipeline_static_first(debug)
        else:
            return self._pipeline_adaptive(debug)
