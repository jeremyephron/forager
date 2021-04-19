import torch
import clip

from typing import List

from knn import utils
from knn.mappers import Mapper

import config


class TextEmbeddingMapper(Mapper):
    def initialize_container(self):
        self.model, _ = clip.load(config.CLIP_MODEL, device="cpu")

    @utils.log_exception_from_coro_but_return_none
    async def process_chunk(
        self, chunk: List[str], job_id, job_args, request_id
    ) -> List[str]:
        text = clip.tokenize(chunk)
        text_features = self.model.encode_text(text).numpy()
        return list(map(utils.numpy_to_base64, text_features))

    async def process_element(self, *args, **kwargs):
        raise NotImplementedError()


app = TextEmbeddingMapper().server
