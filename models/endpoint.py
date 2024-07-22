from pydantic import BaseModel
from typing import Optional, List, Dict

class EmbedRequest(BaseModel):
    text: str
    dense: Optional[bool] = None
    sparse: Optional[bool] = None
    colbert: Optional[bool] = None

class EmbedResponse(BaseModel):
    text: str
    dense: Optional[List[float]]
    colbert: Optional[List[List[float]]]
    sparse: Optional[Dict[str, float]]

    @classmethod
    def from_dict(cls, data) -> "EmbedResponse":
        return cls(**data)