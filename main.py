from fastapi import FastAPI
from pydantic import BaseModel
import json, os

from processing import (
    get_embedding, 
    load_pretrained, 
    preprocessing, 
    download
)

model = load_pretrained()

app = FastAPI(
    title="Recommend API",
)


class URL(BaseModel): 
    url: str

@app.post('/getembedding', tags=['Get Embedding'])
def getembedding(item: URL):
    dir_data= download(item.url)
    if dir_data: 
        mel_spec= preprocessing(dir_data)
        embedding= get_embedding(model, mel_spec, rate= 1)
        os.remove(dir_data)
        return json.dumps(embedding.tolist())

    else: 
        return None  




# from typing import Optional
# from fastapi import FastAPI
# from pydantic import BaseModel # import class BaseModel của thư viện pydantic


# class Item(BaseModel): # kế thừa từ class Basemodel và khai báo các biến
#     name: str
#     description: Optional[str] = None
#     price: float
#     tax: Optional[float] = None


# app = FastAPI()


# @app.post("/items/")
# async def create_item(item: Item): # khai báo dưới dạng parameter
#     return item