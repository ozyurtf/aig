from pydantic import BaseModel, Field

class YearExtraction(BaseModel):
    years: list[str] = Field(default_factory=list)