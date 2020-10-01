from pathlib import Path
from core.detectors import detector_create
from core import Video
from config import *

import cv2

with Video(filename) as video:
    detector = detector_create(detector_key, video=video, kind=kind)

    pipeline, preprocessing = detector.get_pipeline(debug=debug), None

    video.current_frame_position = start_from
    frame = video.get_frame(preprocessing=preprocessing)

    while frame is not None:

        out = pipeline.exec(frame)

        if display_video:
            if isinstance(out, (list, tuple)):
                for index, item in enumerate(out):
                    cv2.imshow(f"frame{index}", item)
            else:
                cv2.imshow("frame", out)

            cv2.waitKey(10)

        frame = video.get_frame(preprocessing=preprocessing)
