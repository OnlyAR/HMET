from pydantic import BaseModel


class Query(BaseModel):
    id: int
    query: str


class Candidate(BaseModel):
    id: int
    content: str
    fact: str
    reason: str
    result: str
