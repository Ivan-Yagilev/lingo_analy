from fastapi import FastAPI
from use import use_model
from models import Input

app = FastAPI()

@app.get("/")
def emotion(input: Input):
    return use_model(input.input_text)
