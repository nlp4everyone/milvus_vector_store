from pydantic import BaseModel
from typing import (List,
                    Dict,
                    Any,
                    Union,
                    Optional)

class FundamentalField(BaseModel):
    id_ :str
    embedding :List[float]
    sparse_embedding :Any

class BaseNodeField(BaseModel):
    id_ :str
    metadata :Dict[str,Any]
    excluded_embed_metadata_keys :List[str]
    excluded_llm_metadata_keys :List[str]
    relationships :Dict
    text :str
    mimetype :str
    start_char_idx :Optional[int] = None
    end_char_idx :Optional[int] = None
    text_template :str
    metadata_template :str
    metadata_seperator :str

class DocumentField(BaseModel):
    metadata :Dict
    page_content :str