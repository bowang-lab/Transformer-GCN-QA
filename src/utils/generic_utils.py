import errno
import os


# https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist#273227
def make_dir(directory):
    """Creates a directory at `directory` if it does not already exist.
    """
    try:
        os.makedirs(directory)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
