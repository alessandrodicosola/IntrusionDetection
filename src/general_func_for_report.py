# ===============================================================================
# === THIS FILES CONTAINS FUNCTIONS USED FOR GENERATING IMAGES FOR THE REPORT ===
# ===============================================================================


import cv2
from core.common import get_diff, display_images
import matplotlib.pyplot as plt
import numpy as np
from distances import l1, l2, max_distance


def generate_mask_distances(video, threshold_range, num_frames=2, preprocessing=None):
    # amount of thresholds
    threshold_list = list(threshold_range)
    distances = [l1, l2, max_distance]
    fig, axes = plt.subplots(len(threshold_list), len(distances), figsize=(10, 20))
    frames = video.get_frames(num_frames)
    for c, d in enumerate(distances):
        for r, t in enumerate(threshold_list):
            mask = get_diff(frames, distance=d, threshold=lambda diff: diff > t)
            mask = np.logical_not(mask).astype(np.uint8)
            axes[r, c].imshow(mask, cmap='gray')
            axes[r, c].set_title(f"distance={d.__name__} threshold={t}")

    plt.show()


def generate_mask_examples(video, frames_range, threshold_range, distance, preprocessing=None):
    # amount of frames
    frames_list = list(frames_range)
    # amount of thresholds
    threshold_list = list(threshold_range)

    fig, axes = plt.subplots(len(frames_list), len(threshold_list), figsize=(10, 20))

    for n in range(len(frames_list)):
        # since n goes from 0 to len(frames_list) we have to add frames_range.start
        real_n = n + frames_range.start

        frames = video.get_frames(real_n, preprocessing)

        for t in range(len(threshold_list)):
            # same as before: we want to start from 1 and add threshold_range.step each time
            real_t = (t + 1) * threshold_range.step
            # object detected moving are white
            mask = get_diff(frames, distance=distance, threshold=lambda diff: diff > real_t)
            # object detected moving are now black
            mask = np.logical_not(mask)
            axes[n, t].imshow(mask.astype(np.uint8) * 255, cmap='gray', vmin=0, vmax=255)
            axes[n, t].set_title(f"frames={real_n} threshold={real_t}")
    plt.tight_layout()
    plt.suptitle(
        f"using {distance.__name__} distance" + (f" and {preprocessing.__name__}" if preprocessing is not None else ""))
    plt.show()


def check_thresholding(frame):
    val_thr = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
    val_thr = [thr + cv2.THRESH_OTSU for thr in val_thr]

    images = [cv2.threshold(frame, 0, 255, thr)[1] for thr in val_thr]
    display_images(images)


from core import Video, Pipeline
from core.common import get_grad


def gen_pipeline(video: Video):
    pipeline = Pipeline(debug=True)
    frame = video.get_frame()

    # Show the input
    pipeline.add_operation("Input", lambda frame: frame)

    pipeline.add_operation("Gradient", lambda frame: get_grad(frame))

    #pipeline.add_operation("Heatmap", lambda frame: cv2.applyColorMap(frame, cv2.COLORMAP_JET))

    pipeline.add_operation("Mask", lambda frame: frame < 30)

    def showonlymask(mask):
        import numpy as np
        input = pipeline.input.copy()
        out = np.empty_like(input)
        out[mask] = input[mask]
        return out

    pipeline.add_operation("Apply mask on the input image", showonlymask)

    result = pipeline.exec(frame)

    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    with Video(R"..\\in\\rilevamento-intrusioni-video.avi") as video:
        gen_pipeline(video)
