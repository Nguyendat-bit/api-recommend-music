import os 
import gdown 
import requests
import torch 
import librosa 
import json
import numpy as np 
from model import AudioClassify_model
from contant import model_url, num_classes, input_shape

def load_pretrained(): 
    if not os.path.isfile('pretrained/resnet_audio.pth'):
        print('Install Pretrained!')
        gdown.download(id= model_url, output= 'pretrained/resnet_audio.pth', fuzzy= True)
    
    print('Load Pretrained!')
    model= AudioClassify_model(input_shape, num_classes)
    model.load_state_dict(torch.load('pretrained/resnet_audio.pth'))
    # model = nn.Sequential(*list(model.children())[:-1])

    model.eval() 

    return model 

def preprocessing(dir, sr= 44100): 

    frame_length= int(sr * 0.3)
    hop_length= int(frame_length * 0.5) 

    signal, sr= librosa.load(dir, sr= sr)
    signal= signal[: sr * 30]

    frames = librosa.util.frame(signal, frame_length= frame_length, hop_length= hop_length).T

    mel_spec= librosa.feature.melspectrogram(y= frames, sr= sr, n_mels= 128) 
    
    mel_spec= librosa.power_to_db(mel_spec, ref= np.max) 

    return mel_spec 

def get_embedding(model, x, rate= 1): 
    standardized= lambda x: (x - x.min()) / (-1 * x.min()) * 2 -1
    
    x = list(map(standardized, x))

    # print(x)
    x = torch.tensor(np.array(x)).unsqueeze(1)
    print(x.shape)
    with torch.no_grad(): 
        output= model(x)
    
    return output[: int(rate * len(x))].view(-1)

def download(link):
    response= requests.get(link)
    if response.status_code == 200: 
        filename= os.path.basename(link)
        print(f'Installing {filename}')
        with open('data/' + filename, 'wb') as f:
            f.write(response.content)

        return 'data/' + filename
    else: 
        print('Cannot Install')
        return None

def call_api_get_embedding():
    url= 'https://0c03-2405-4802-1d7e-2d20-17e1-a685-5a1-4c4f.ngrok-free.app/api/admin/embedding?page=1&limit=10000'
    response= requests.get(url)
    if response.status_code == 200: 
        print('Query all embedding') 

        data= json.loads(response.content)['data']
        get_musicId = lambda x: x['musicId']
        get_embedding= lambda x: json.loads(x['embedding'])

        musicIDs= list(map(get_musicId, data))
        embeddings= list(map(get_embedding, data))

        return musicIDs, embeddings
    else: 
        print('Cannot query all embedding')
        return None, None

