from core.Video import Video


class Detector:
    def __init__(self, video: Video):
        """
        Init detector
        :param video: Video object
        """
        self.video = video

    def get_pipeline(self, debug=False):
        pass

    def get_preprocessing(self):
        pass
