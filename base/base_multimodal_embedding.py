# Typing
from typing import List, Union
# PIL Image
from PIL import Image

class BaseMultimodalEmbedding:
    @property
    def model_name(self) -> str:
        raise NotImplementedError()

    def embed_documents(self,
                        texts :Union[str,List[str]],
                        **kwargs) -> List[List[float]]:
        raise NotImplementedError()

    def embed_images(self,
                     images :List[Union[str,Image.Image]],
                     **kwargs) -> List[List[float]]:
        raise NotImplementedError()
