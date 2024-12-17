from pydantic import BaseModel
from typing import (List,Literal)

class FundamentalField(BaseModel):
    id_ :str
    embedding :List[float]
    sparse_embedding :List[dict]
    class_name :Literal["Document","TextNode","ImageNode"]

