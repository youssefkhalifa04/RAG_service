from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, documents):
        pass

    @abstractmethod
    def similarity(self, query_vec, doc_vecs):
        pass
    @abstractmethod
    def get_model_id(self) -> str:
        pass
    @abstractmethod
    def getResponse(self, messages):
        pass
    