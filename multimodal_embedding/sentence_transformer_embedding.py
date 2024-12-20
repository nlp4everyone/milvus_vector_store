# Typing
from typing import List, Union
# Inheritance
from ..base import BaseMultimodalEmbedding
# PIL Image
from PIL import Image
# Util
# Sentence transformer
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding(BaseMultimodalEmbedding):
    def __init__(self,
                 image_model_name :str = "clip-ViT-B-32",
                 text_model_name :str = "clip-ViT-B-32-multilingual-v1",
                 multilingual :bool = False,
                 **kwargs):
        # Init text model
        self._image_model = SentenceTransformer(model_name_or_path = image_model_name,
                                                **kwargs)
        if not multilingual:
            self._text_model = self._image_model
        else:
            # Init image model
            self._text_model = SentenceTransformer(model_name_or_path = text_model_name,
                                                   **kwargs)

    def embed_documents(self,
                        texts :Union[str,List[str]],
                        batch_size: int = 8,
                        **kwargs) -> List[List[float]]:
        # When texts is string
        if isinstance(texts,str): texts = [texts]
        # Embed documents
        return self._text_model.encode(sentences = texts,
                                       batch_size = batch_size,
                                       **kwargs).tolist()
    def embed_images(self,
                     images,
                     batch_size: int = 8,
                     **kwargs) -> List[List[float]]:
        # Embedding
        return self._image_model.encode(images,
                                        batch_size = batch_size,
                                        **kwargs).tolist()
