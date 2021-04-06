import requests
import time

import config


def download(
        self, image_path: str,
        num_retries: int = config.DOWNLOAD_NUM_RETRIES) -> bytes:
    for i in range(num_retries + 1):
        try:
            response = requests.get(image_path)
            assert response.status_code == 200
            return response.content
        except Exception:
            if i < num_retries:
                time.sleep(2 ** i)
            else:
                raise
    assert False  # unreachable
