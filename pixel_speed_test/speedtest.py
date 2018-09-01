import time
from functools import wraps

import numpy as np
from scipy import ndimage
import cv2


TIMES_TO_RUN = 5


def np_shift_v(arr, num, fill_value=None):
    result = np.empty_like(arr)
    if num > 0:
        result[:num, :] = fill_value
        result[num:, :] = arr[:-num, :]
    elif num < 0:
        result[num:, :] = fill_value
        result[:num, :] = arr[-num:, :]
    else:
        result = arr
    return result


def np_shift_h(arr, num, fill_value=None):
    result = np.empty_like(arr)
    if num > 0:
        result[:, :num] = fill_value
        result[:, num:] = arr[:, :-num]
    elif num < 0:
        result[:, num:] = fill_value
        result[:, :num] = arr[:, -num:]
    else:
        result = arr
    return result


def quad_neighbor_pixels_from_ndarray(image, pixel_class):
    pixels = pixels_from_ndarray(image, pixel_class)
    # pixel_quad_neighbors_from_ndarray(pixels)  # <- attribute-access modification
    return pixels


def pixels_from_ndarray(image, pixel_class, pixel_data_type=None):
    if pixel_data_type is None:
        pixel_data_type = np.dtype(pixel_class)
    vertical_indices, horizontal_indices = np.indices(image.shape)
    return _iter_convert_pixels(
        image, vertical_indices, horizontal_indices, pixel_data_type,
        lambda pixel_value, vertical_index, horizontal_index:
        pixel_class(pixel_value, (vertical_index, horizontal_index))
    )


def _iter_convert_pixels(image_to_iter, vertical_indices, horizontal_indices, pixel_data_type, pixel_from_attributes):
    pixel_convert = np.frompyfunc(pixel_from_attributes, 3, 1)
    convert_iterable = np.nditer([image_to_iter, vertical_indices, horizontal_indices, None],
                                 flags=['external_loop', 'buffered', 'refs_ok'],
                                 op_dtypes=['uint8', 'int', 'int', pixel_data_type],
                                 op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly', 'allocate']])
    for pixel_value, vertical_location, horizontal_location, this_output in convert_iterable:
        this_output[...] = pixel_convert(pixel_value, vertical_location, horizontal_location)
    return convert_iterable.operands[3]


def pixel_quad_neighbors_from_ndarray(pixel_ndarray):
    _assign_ndarray_quad_neighbors(pixel_ndarray, *_get_ndarray_quad_neighbors(pixel_ndarray))


def _get_ndarray_quad_neighbors(pixel_ndarray):
    up_neighbors = np_shift_v(pixel_ndarray, 1)
    left_neighbors = np_shift_h(pixel_ndarray, 1)
    down_neighbors = np_shift_v(pixel_ndarray, -1)
    right_neighbors = np_shift_h(pixel_ndarray, -1)
    return up_neighbors, left_neighbors, down_neighbors, right_neighbors


def _assign_ndarray_quad_neighbors(pixel_ndarray, up_neighbors, left_neighbors, down_neighbors, right_neighbors):
    np_set_neighbors = np.frompyfunc(_set_pixel_neighbors, 5, 0)
    np_set_neighbors(pixel_ndarray, up_neighbors, left_neighbors, down_neighbors, right_neighbors)


def _set_pixel_neighbors(pix, *neighbors):
    """
    Create a tuple of references to neighbors in pix.

    :Parameters:
        :param Pixel pix: Pixel object to set neighbor references within.
        :param Pixels tuple neighbors: The (up, left, down, right) Pixel neighbors of pix.
    :rtype: None
    :return: None
    """
    pix.neighbors = neighbors


def binary_label_ndarray(image):
    # White Shapes label
    white_labeled, white_labels = label_ndarray_ones(image)

    # Black Shapes label
    black_labeled, black_labels = label_ndarray_ones(1 - image)

    return black_labeled, black_labels, white_labeled, white_labels


def label_ndarray_ones(image, *np_args, **np_kwargs):
    """
    Segment and label the regions of connected 1's in an image.

    :Parameters:
        :param numpy.ndarray image: The image to be labeled.
        :param np_args: (Optional) Positional arguments provided to ndimage.label after image.
        :param np_kwargs: (Optional) Keyword arguments provided to ndimage.label after np_args.
    :rtype: numpy.ndarray, numpy.ndarray
    :return: image array with ones converted to the label group id, numpy.ndarray with a label id value per index.
    """
    image_labeled, number_labels = ndimage.label(image, *np_args, **np_kwargs)
    labels = np.arange(1, number_labels + 1)
    return image_labeled, labels


def binary_shapes_from_labels(pixels,
                              black_labeled, black_labels,
                              white_labeled, white_labels,
                              shape_class, *args, shape_data_type=None, **kwargs):
    if shape_data_type is None:
        shape_data_type = np.dtype(shape_class)

    black_shapes = shapes_from_labels(
        pixels, black_labeled, black_labels, shape_class, shape_data_type, *args, **kwargs
    )

    white_shapes = shapes_from_labels(
        pixels, white_labeled, white_labels, shape_class, shape_data_type, *args, **kwargs
    )

    return black_shapes, white_shapes


def shapes_from_labels(pixels, labeled_image, labels, shape_class, shape_data_type, *args, **kwargs):
    return ndimage.labeled_comprehension(
        pixels, labeled_image, labels,
        lambda shape_pixels: shape_class(shape_pixels, *args, **kwargs),
        shape_data_type, None, False
    ).tolist()


class Pixel(object):
    """Object representing individual pixel values and indices in an image."""

    __slots__ = ['color', 'coordinates', 'shape', 'neighbors', 'navigation_pixel']

    def __init__(self, color, coordinates):
        """
        "Set color and indices from source image."

        :Parameters:
            :param int color: The color of the thresholded source image pixel, should be either 1 or 0.
            :param ints tuple coordinates: The (column, row) index of the source image pixel.
        :rtype: None
        :return: None
        """
        self.color = color
        self.coordinates = coordinates
        self.neighbors = (None, None, None, None)
        self.shape = None
        self.navigation_pixel = None

    def set_neighbors(self, *neighbors):
        self.neighbors = neighbors


class Shape(object):
    """A continuous group of same-color Pixel objects representing a connected object."""

    __slots__ = ['row', 'column', 'segment', 'inner', 'owned', 'roots', 'pixels', 'color',
                 '_coordinates', '_area', '_height', '_width', '_box']

    def __init__(self, init_pixels):
        """
        Set reference to owned pixel objects responsible for creation and the source space reference.

        :Parameters:
            :param Pixels ndarray or list init_pixels: The Pixels responsible for the creation of this object.
        :rtype: None
        :return: None
        """
        self.row = None
        self.column = None
        self.segment = None
        self.inner = set()
        self.owned = set()
        self.roots = {0: None, 1: None}
        self._coordinates = None
        self._area = None
        self._height = None
        self._width = None
        self._box = None
        if init_pixels is not None:
            if hasattr(init_pixels, "tolist"):
                # self.pixels = set(init_pixels.tolist())  # <- attribute-access modification
                self.pixels = init_pixels  # <- attribute-less modification
            else:
                # self.pixels = set(init_pixels)  # <- attribute-access modification
                self.pixels = init_pixels  # <- attribute-less modification
            # self.color = next(iter(self.pixels)).color if self.pixels else None  # <- attribute-access modification
            self.color = None  # <- attribute-less modification
        else:
            self.pixels = set()
            self.color = None
        # self.assign_pixels(self.pixels)  # <- attribute-access modification

    def assign_pixels(self, pixels):
        for pix in pixels:
            pix.shape = self


class DictPixel(object):
    """Object representing individual pixel values and indices in an image."""

    # __slots__ = ['color', 'coordinates', 'shape', 'neighbors', 'navigation_pixel']

    def __init__(self, color, coordinates):
        """
        "Set color and indices from source image."

        :Parameters:
            :param int color: The color of the thresholded source image pixel, should be either 1 or 0.
            :param ints tuple coordinates: The (column, row) index of the source image pixel.
        :rtype: None
        :return: None
        """
        self.color = color
        self.coordinates = coordinates
        self.neighbors = (None, None, None, None)
        self.shape = None
        self.navigation_pixel = None

    def set_neighbors(self, *neighbors):
        self.neighbors = neighbors


class DictShape(object):
    """A continuous group of same-color Pixel objects representing a connected object."""

    # __slots__ = ['row', 'column', 'segment', 'inner', 'owned', 'roots', 'pixels', 'color',
    #              '_coordinates', '_area', '_height', '_width', '_box']

    def __init__(self, init_pixels):
        """
        Set reference to owned pixel objects responsible for creation and the source space reference.

        :Parameters:
            :param Pixels ndarray or list init_pixels: The Pixels responsible for the creation of this object.
        :rtype: None
        :return: None
        """
        self.row = None
        self.column = None
        self.segment = None
        self.inner = set()
        self.owned = set()
        self.roots = {0: None, 1: None}
        self._coordinates = None
        self._area = None
        self._height = None
        self._width = None
        self._box = None
        if init_pixels is not None:
            if hasattr(init_pixels, "tolist"):
                # self.pixels = set(init_pixels.tolist())  # <- attribute-access modification
                self.pixels = init_pixels  # <- attribute-less modification
            else:
                # self.pixels = set(init_pixels)  # <- attribute-access modification
                self.pixels = init_pixels  # <- attribute-less modification
            # self.color = next(iter(self.pixels)).color if self.pixels else None  # <- attribute-access modification
            self.color = None  # <- attribute-less modification
        else:
            self.pixels = set()
            self.color = None
        # self.assign_pixels(self.pixels)  # <- attribute-access modification

    def assign_pixels(self, pixels):
        for pix in pixels:
            pix.shape = self


def nanotime(_wrapped_func, *, repeat=TIMES_TO_RUN):
    nano_timer = time.clock

    def nanodecorator(wrapped_func):
        @wraps(wrapped_func)
        def nanowrapper_with_blockchain_hyperfabric():
            pre_time = nano_timer()
            for _ in range(repeat):
                wrapped_func()
            post_time = nano_timer()
            print("Took {} seconds to run {} {} times..".format(post_time - pre_time, wrapped_func, repeat))
            return
        return nanowrapper_with_blockchain_hyperfabric
    if _wrapped_func is None:
        return nanodecorator
    return nanodecorator(_wrapped_func)


@nanotime
def timeable_conversion_all_slots():
    test_im = "./test.png"
    import_img = cv2.imread(test_im, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(import_img, thresh=122, maxval=1, type=cv2.THRESH_BINARY)

    # Shapes labels
    black_labeled, black_labels, white_labeled, white_labels = binary_label_ndarray(image)

    # Init Pixels; then stack and assign pixel neighbors.
    np_pixels = quad_neighbor_pixels_from_ndarray(image, Pixel)

    # Extract objects/pixels to lists
    shapes_0, shapes_1 = binary_shapes_from_labels(
        np_pixels, black_labeled, black_labels, white_labeled, white_labels, Shape
    )
    pixels = np_pixels.tolist()
    return


@nanotime
def timeable_conversion_no_slots():
    test_im = "./test.png"
    import_img = cv2.imread(test_im, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(import_img, thresh=122, maxval=1, type=cv2.THRESH_BINARY)

    # Shapes labels
    black_labeled, black_labels, white_labeled, white_labels = binary_label_ndarray(image)

    # Init Pixels; then stack and assign pixel neighbors.
    np_pixels = quad_neighbor_pixels_from_ndarray(image, DictPixel)

    # Extract objects/pixels to lists
    shapes_0, shapes_1 = binary_shapes_from_labels(
        np_pixels, black_labeled, black_labels, white_labeled, white_labels, DictShape
    )
    pixels = np_pixels.tolist()
    return


@nanotime
def timeable_conversion_slot_pixel():
    test_im = "./test.png"
    import_img = cv2.imread(test_im, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(import_img, thresh=122, maxval=1, type=cv2.THRESH_BINARY)

    # Shapes labels
    black_labeled, black_labels, white_labeled, white_labels = binary_label_ndarray(image)

    # Init Pixels; then stack and assign pixel neighbors.
    np_pixels = quad_neighbor_pixels_from_ndarray(image, Pixel)

    # Extract objects/pixels to lists
    shapes_0, shapes_1 = binary_shapes_from_labels(
        np_pixels, black_labeled, black_labels, white_labeled, white_labels, DictShape
    )
    pixels = np_pixels.tolist()
    return


@nanotime
def timeable_conversion_slot_shape():
    test_im = "./test.png"
    import_img = cv2.imread(test_im, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(import_img, thresh=122, maxval=1, type=cv2.THRESH_BINARY)

    # Shapes labels
    black_labeled, black_labels, white_labeled, white_labels = binary_label_ndarray(image)

    # Init Pixels; then stack and assign pixel neighbors.
    np_pixels = quad_neighbor_pixels_from_ndarray(image, DictPixel)

    # Extract objects/pixels to lists
    shapes_0, shapes_1 = binary_shapes_from_labels(
        np_pixels, black_labeled, black_labels, white_labeled, white_labels, Shape
    )
    pixels = np_pixels.tolist()
    return


def main():
    timeable_conversion_all_slots()
    timeable_conversion_slot_pixel()
    timeable_conversion_slot_shape()
    timeable_conversion_no_slots()


def reverse_main():
    timeable_conversion_no_slots()
    timeable_conversion_slot_shape()
    timeable_conversion_slot_pixel()
    timeable_conversion_all_slots()


if __name__ == "__main__":
    main()
    # reverse_main()
