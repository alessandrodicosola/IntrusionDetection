def get_diff(frames, distance, threshold):
    import numpy as np
    """
    Compute the difference between frames
    :param frames: np.array of frames
    :param distance: distance function: def distance(frame1,frame2)
    :param threshold: thresholding function. def thr(diff)
    :return: np.array of the difference between images computed (as type np.uint8)
    """
    # compute the first differences with threshold
    diff = [(distance(frames[i - 1], frames[i])) for i in reversed(range(1, len(frames)))]
    if threshold is not None: diff = [threshold(img) for img in diff]

    # << and >> all differences until the result is obtained
    while len(diff) > 1:
        diff = [np.prod([diff[i - 1], diff[i]], axis=0) for i in reversed(range(1, len(diff)))]

    assert (len(diff) == 1)
    return diff[0].astype(np.uint8)


def gamma_correction(image, gamma):
    import numpy as np
    import cv2
    """
    Apply gamma correction on the image for increase (gamma < 1) or decrease (gamma > 1) brightness of the image
    :param image: 
    :param gamma: 
    :return: image corrected
    """
    return 255 * (1 / 255 * image) ** gamma


def display_images(images, titles=None, max_col=0):
    """
    Display images in subplots
    :param images: list of images
    :param titles: list of titles
    :param max_col: maximum amount of columns (default=0: all images in a row)
    """
    import matplotlib.pyplot as plt
    if len(images) > 1:
        n_col = len(images) if max_col == 0 else max_col
        n_row = len(images) // n_col + (0 if len(images) % n_col == 0 else 1)
        fig, axes = plt.subplots(n_row, n_col, figsize=(10, 20))

        for index, ax in enumerate(axes.flat):
            ax.axis('off')
            if index < len(images) and images[index] is not None:
                ax.imshow(images[index], cmap="gray")
                if titles is not None and index < len(titles) and titles[index] is not None:
                    ax.set_title(titles[index])
    else:
        if images is not None: plt.imshow(images[0], cmap="gray")
        if titles is not None: plt.title(titles[0])
        plt.axis('off')

    plt.show()


def find_contours(mask, filter=None):
    """
    Method for finding and filtering contours.
    :param mask: binary image where to find contours
    :param filter: filter function. filter(contour) -> True: accepted; False: rejected
    :return: list of filtered contours as class core.Contour.Contour
    """
    import cv2
    from core import Contour
    # RETR_EXTERNAL: retrieves only the extreme outer contours: so no child contours.
    # CHAIN_APPROX_NONE: retrieves all points of the contour
    contours, hierarchies = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [Contour(contour) for contour in contours if (filter(contour) if filter is not None else True)]


def filter_contours(raw_contour):
    """
    Basic filtering function.
    Return True for big objects or with meaningless area and perimeter.
    :param raw_contour: contour points
    :return: bool
    """
    import cv2
    area = cv2.contourArea(raw_contour)
    perimeter = cv2.arcLength(raw_contour, closed=True)
    x, y, w, h = cv2.boundingRect(raw_contour)
    rectangularity = area / w * h

    if rectangularity > 5000:
        return True
    else:
        if perimeter > 100 and area > 300:
            return True

    # by default
    return False


def draw_contours(contours, frame_input):
    """
    Draw contours
    :param contours: list of contours
    :param frame_input: input
    :return:
    """
    import numpy as np
    import matplotlib.cm as cm
    import cv2

    # Prepare the output image
    out_no_alpha = frame_input.copy()
    # Out where we want some alpha
    out_alpha = out_no_alpha.copy()

    # Return the Pastel1 color map using an alpha channel and values as integer 0-255 (so bytes)
    alpha = 0.2
    cmap = cm.get_cmap("Pastel1")
    max_colors = 10
    colors = cmap(np.linspace(0, 1, max_colors), alpha=alpha, bytes=True)
    colors = [(int(color[0]), int(color[1]), int(color[2]), int(color[3])) for color in colors]

    # In case of no contours
    out = out_no_alpha

    for index, cnt in enumerate(contours):
        # TEXT
        y = max(0, cnt.y - 10)
        cv2.putText(out_no_alpha, f"[{index}]{cnt.label}", (cnt.x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Avoid out of bounds
        index = index % max_colors
        color = colors[index][:3]

        # Fill contour using alpha
        cv2.fillPoly(out_alpha, [cnt.contour], color)
        # Draw contour without alpha
        cv2.drawContours(out_no_alpha, [cnt.contour], 0, color, 2)

        out = cv2.addWeighted(out_alpha, alpha, out_no_alpha, 1 - alpha, 0)

    return out


def get_grad(image):
    """
    Get the gradient of the image inside the bounding box
    :param image: image to which compute the gradient
    :return:
    """
    import numpy as np
    import cv2
    central_diff_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    central_diff_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    # compute derivatives
    dx = np.abs(cv2.filter2D(image, -1, central_diff_x))
    dy = np.abs(cv2.filter2D(image, -1, central_diff_y))
    grad = np.maximum(dx, dy)
    return grad


def check_args(dictionary: dict):
    from functools import wraps
    import inspect
    """
    Decorator for checking arguments of a function
    :param dictionary: dictionary key: choices 
    :raises ValueError if the argument {key} has no values in {choices}
    """

    # Static check
    # argument is a dictionary
    assert isinstance(dictionary, dict)
    # each choices element is a iterable
    for key in dictionary:
        assert isinstance(dictionary[key], (tuple, list, dict))

    def check_arg_internal(init):
        @wraps(init)
        def init_wrapper(*args, **kwargs):
            sham_bind = inspect.signature(init).bind(*args, **kwargs)
            for arg in sham_bind.arguments:
                # skip argument not specified inside the dictionary
                if arg not in dictionary: continue
                if sham_bind.arguments[arg] not in dictionary[arg]:
                    raise ValueError(f"{arg} must be in {dictionary[arg]}")
            return init(*args, **kwargs)

        return init_wrapper

    return check_arg_internal
