import pydantic

class Input(pydantic.BaseModel):
    input_text: str

class Output(pydantic.BaseModel):
    prediction: str
    score: float