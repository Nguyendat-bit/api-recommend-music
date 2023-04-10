from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from pyngrok import ngrok
import uvicorn

from pydantic import BaseModel
import json, os


os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


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

app.add_middleware(
    CORSMiddleware,
    allow_origins= ['*'],
    allow_credentials= True, 
    allow_methods= ['*'],
    allow_headers= ['*'],
)

class URL(BaseModel): 
    url: str
    rate: float


@app.post('/getembedding', tags=['Get Embedding'])
def getembedding(item: URL):
    dir_data= download(item.url)
    if dir_data: 
        mel_spec= preprocessing(dir_data)
        embedding= get_embedding(model, mel_spec, rate= item.rate)
        os.remove(dir_data)
        return json.dumps(embedding.tolist())


# specify a port
port = 8000
ngrok_tunnel = ngrok.connect(port, bind_tls=True)

# where we can visit our fastAPI app
print('Public URL:', ngrok_tunnel.public_url)


nest_asyncio.apply()