from os.path import join, dirname, abspath

from .log import logger

utils_dir = dirname(abspath(__file__))
root_dir = join(utils_dir, "..")
data_dir = join(root_dir, "data")
output_dir = join(root_dir, "output")
resources_dir = join(utils_dir, 'resources')
