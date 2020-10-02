import cv2


class Contour:
    def __init__(self, contour):
        """
        Initialize Contour
        :param contour: list of points got with cv2.findContours
        """
        self.contour = contour

        # Init properties
        self.area = cv2.contourArea(self.contour)
        self.perimeter = cv2.arcLength(self.contour, closed=True)
        self.x, self.y, self.w, self.h = cv2.boundingRect(self.contour)
        self.ratio = self.w / self.h
        self.circularity = self.area / (3.14 / 4 * self.w * self.h)
        self.rectangularity = self.area / self.w * self.h

        self.std = -1
        self.mean = -1

        self.label = ""

    def get_label(self, input_frame, threshold):
        """
        Compute the label
        """
        label = ""
        # 0.3: height three times width
        if self.area > 3000 or self.ratio < 0.3:
            label = "person"
        else:
            label = "other"
        self.label = label + (" false positive" if self.false_positive(input_frame, threshold) else "")

    def false_positive(self, input_frame, threshold):
        """
        :param input_frame: input image necessary for computing false positiveness
        :param threshold: threshold for false positiveness. Between 0 and 1
        :return: True if false positive
        """
        import numpy as np
        from core.common import get_grad
        # capture the image inside the bounding box of the contour
        grad = get_grad(self.get_image(input_frame, use_bounding_box=True))
        # normalize over 255
        grad_norm = grad / 255
        mean, std = np.mean(grad_norm), np.std(grad_norm)
        self.mean = mean
        self.std = std
        return std < threshold

    def get_image(self, input_frame, use_bounding_box=False):
        """
        Get the image inside the contour or the bounding box of the contour
        :param input_frame: input image
        :param use_bounding_box Use the image inside the bounding box
        :return:
        """
        if use_bounding_box:
            return input_frame[self.y:self.y + self.h, self.x:self.x + self.w]
        else:
            return self._get_image_inside_contour(input_frame)

    def _get_image_inside_contour(self, input_frame):
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        # Init output
        out = np.zeros_like(input_frame)
        # Init background with black pixels
        bg = out.copy()
        # Draw the contour filled (thickness=-1)
        mask = cv2.drawContours(bg, [self.contour], 0, 255, thickness=-1)
        # Copy pixels from the input image to the output image where pixels intensity is 255 in the mask
        out[mask == 255] = input_frame[mask == 255]
        return out[self.y:self.y + self.h, self.x:self.x + self.w]


    def get_info(self):
        # acquire information from the ones specified below
        return [str(getattr(self, name, "")) for name in Contour.get_info_names()]

    @staticmethod
    def get_info_names():
        return ["area", "perimeter", "ratio", "circularity", "rectangularity", "mean", "std", "label"]

    def get_log(self, index):
        log = str(index) + "\t"
        log += "\t".join(self.get_info())
        return log
