import gzip

IMAGE_SIZE = 28
NUM_CHANNELS = 1
import numpy


def extract_data(filename, num_images):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)

    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS
        )
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (255 / 2.0)) / 255
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

        return data


def extract_labels(filename, num_images):
    """
    Extract the labels into a vector of int64 label IDs.
    """
    print('Extracting labels', filename)
    with gzip.open(filename) as bytestream:
        # Discard header.
        bytestream.read(8)
        # Read bytes for labels.
        buf = bytestream.read(num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels
