from collections import namedtuple
from core.common import display_images
from copy import deepcopy


class Pipeline:
    """
    Base class for handling the sequence of transformations or operations
    """
    Operation = namedtuple("Operation", "Name Function Hide")

    def __init__(self, debug=False):
        """
        :param debug: if True a matplotlib figure with subplots will be show all operations not hidden
        """
        self.debug = debug
        self._operations = list()
        self.input = None
        self._custom_objects = {}

    def store(self, key, value):
        """
        Save a custom object in the pipeline in order to be able to load it after
        :param key: key
        :param value: value
        """
        self._custom_objects[key] = value

    def load(self, key):
        """
        Retrieve a custom object from the internal store
        :param key: key
        :return: value
        """
        return self._custom_objects.get(key)

    def _apply(self, image):
        """
        Apply operations sequentially
        :param image: starting image
        :return: image transformed
        """
        result = image

        if self.debug:
            images_to_display = []
            titles = []

        for operation in self._operations:
            result = operation.Function(result)

            if self.debug:
                if not operation.Hide:
                    if isinstance(result, (list, tuple)):
                        images_to_display.extend(deepcopy(result))
                        titles.extend(operation.Name)
                    else:
                        images_to_display.append(result.copy())
                        titles.append(operation.Name)

        if self.debug:
            display_images(images_to_display, titles=titles, max_col=3)
            # free memory
            images_to_display.clear()
            titles.clear()

        return result

    def add_operation(self, name, function, hide=False):
        """
        Add an operation to the pipeline
        :param name: name of the operation. It will be displayed as title in subplot
        :param function: function to use: func(prev_output)
        :param hide: hide the output from the subplots
        :return: operation(name,function)
        """
        self._operations.append(self.Operation(name, function, hide))
        return self

    def exec(self, to_image):
        """
        Exec the pipeline
        :param to_image: starting image
        """
        self.input = to_image
        result = self._apply(self.input)

        return result
