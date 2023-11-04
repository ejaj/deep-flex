import os
from urllib import request

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'


def download_data(filename):
    if not os.path.exists(WORK_DIRECTORY):
        os.makedirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)  # Provide both directory and filename
    filepath, _ = request.urlretrieve(SOURCE_URL + filename, filepath)

    size = os.stat(filepath).st_size
    print('Successfully downloaded', filename, size, 'bytes.')
    return filepath
