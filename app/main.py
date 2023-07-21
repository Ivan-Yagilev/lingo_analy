from fastapi import FastAPI
from use import use_model


app = FastAPI()


@app.get("/emotion")
async def root():
    return {"message": "Hello World"}
