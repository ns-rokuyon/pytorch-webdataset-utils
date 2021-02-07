import io
import json
import imghdr
import numpy as np
from typing import Optional, Union
from pathlib import Path


def get_image_extension(img: bytes) -> Optional[str]:
    fp = io.BytesIO(img)
    fp.seek(0)
    ext = imghdr.what(fp)
    # Normalize "jpeg" and "jpg"
    ext = 'jpg' if ext == 'jpeg' else ext
    return ext


def read_image_binary(path: Union[Path, str]) -> bytes:
    with open(path, 'rb') as fp:
        data = fp.read()
    return data


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, x):
        if isinstance(x, np.integer):
            return int(x)
        elif isinstance(x, np.floating):
            return float(x)
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, np.bool_):
            return bool(x)
        return json.JSONEncoder.default(self, x)
