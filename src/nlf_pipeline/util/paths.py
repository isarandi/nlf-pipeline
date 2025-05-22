import os

try:
    DATA_ROOT = os.environ['DATA_ROOT']
except KeyError:
    raise KeyError(
        'The DATA_ROOT environment variable is not set. '
        'Set it to the parent dir of the dataset directories.') from None


try:
    INFERENCE_ROOT = os.environ['INFERENCE_ROOT']
except KeyError:
    raise KeyError(
        'The INFERENCE_ROOT environment variable is not set. '
        'See the readme for how to set it.') from None

