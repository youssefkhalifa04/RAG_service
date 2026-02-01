from interfaces.Storage import Storage
from interfaces.EmbeddingModel import EmbeddingModel
import numpy as np 

class DataEncoder:
    def __init__(self, storage : Storage , embed_model : EmbeddingModel):
        self.storage = storage
        self.embed_model = embed_model

    def encode_data(self, documents, type_label: str, factory_id: str , code: str):
        doc_vecs = self.embed_model.encode(np.array([documents]), prompt_name="data_encoding")
       
        res = self.storage.push_embedding(doc_vecs, factory_id, type_label, documents, code)

        return res