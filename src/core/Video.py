from pathlib import Path
from core.common import check_args

import cv2


class Video:
    """
    Base class for handling simple operation on VideoCapture class
    """

    # Current position of the video file in milliseconds
    _CV_CAP_PROP_POS_MSEC = cv2.CAP_PROP_POS_MSEC
    # 0-based index of the frame to be decoded/captured next.
    _CV_CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    # amount of frames in the video
    _CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{str(self.path.absolute())} does not exist")

    @property
    def current_frame_position(self):
        return self.video.get(self._CV_CAP_PROP_POS_FRAMES)

    @current_frame_position.setter
    def current_frame_position(self, value):
        if value > self.total_frames:
            raise ValueError(f"value must be less than or equal {self.total_frames}")
        self.video.set(self._CV_CAP_PROP_POS_FRAMES, value)

    @property
    def current_frame_msec(self):
        return self.video.get(self._CV_CAP_PROP_POS_MSEC)

    def get_frame(self, preprocessing=None):
        """
        Read a frame
        :param preprocessing: preprocessing function to apply on the frame
        :return: frame with src_type = GRAY as np.array
        """

        ok, frame = self.video.read()
        if not ok:
            pos, msec = self.current_frame_position, self.current_frame_msec
            # if not end of the video
            if pos != self.total_frames:
                raise RuntimeError(f"error retrieving frame {pos} at {msec}")
        # if ret is True and frame is None, end of the video is reached
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
        return frame if preprocessing is None or frame is None else preprocessing(frame)

    def get_frames(self, n, preprocessing=None):
        """
        Read n-frames
        :param preprocessing: preprocessing function on the frame
        :param n: amount of frame to be returned
        :return: list of frames
        """
        return [self.get_frame(preprocessing) for _ in range(n)]

    def get_background(self, interpolation, num_frames, preprocessing=None, start=0):
        """
        Compute the background of the video interpolating first num_frames from start
        :param interpolation: method for interpolating { np.mean, np.median }
        :param num_frames: amount of frames to use
        :param preprocessing: preprocessing function that is used during frame acquisition
        :param start: from where to start getting frame. Default: 0
        :return:
        """
        import numpy as np

        if num_frames + start > self.total_frames:
            raise RuntimeError("Reduce the amount of num_Frames or set start lower")

        current_pos = self.current_frame_position

        self.current_frame_position = start

        def internal_preprocessing(frame):
            if preprocessing is not None: frame = preprocessing(frame)
            return frame.astype(np.float)

        frames = self.get_frames(num_frames, internal_preprocessing)
        bg_interpolated = np.stack(frames, axis=0)
        bg_interpolated = interpolation(bg_interpolated, axis=0)

        # return to the previous frame
        self.current_frame_position = current_pos

        return bg_interpolated.astype(np.uint8)

    def __enter__(self):
        """
        Method used by python when entering the scope of <<with>> keyword
        :return: self
        """
        self.video = cv2.VideoCapture(str(self.path.absolute()))
        if not self.video.isOpened():
            raise RuntimeError(f"error opening the video: {self.path.absolute()}")
        # Init video properties
        self.total_frames = self.video.get(self._CAP_PROP_FRAME_COUNT)

        return self

    def __exit__(self, type, value, traceback):
        """
        Method used by python when exiting the scope of <<with>> keyword
        """
        if self.video.isOpened():
            self.video.release()
            self.video = None
