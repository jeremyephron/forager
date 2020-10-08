from knn.reducers import Reducer


class EmbeddingDictReducer(Reducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = {}

    def handle_result(self, input, output):
        self.embeddings[input] = {
            k: utils.base64_to_numpy(v) for k, v in output.items()
        }

    @property
    def result(self):
        return self.embeddings
