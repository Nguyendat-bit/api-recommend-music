import os 
import gdown 
import requests
import torch 
from torch import nn 
import librosa 
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

    signal, sr= librosa.load(dir)
    signal= signal[: sr * 30]

    frames = librosa.util.frame(signal, frame_length= frame_length, hop_length= hop_length).T

    mel_spec= librosa.feature.melspectrogram(y= frames, sr= sr, n_mels= 128) 
    
    mel_spec= librosa.power_to_db(mel_spec, ref= np.max) 

    return mel_spec 

def get_embedding(model, x, rate= 1): 
    standardized= lambda x: (x - x.min()) / (-1 * x.min()) * 2 -1
    
    x = map(standardized, x)

    x = torch.from_numpy(x)
    with torch.no_grad(): 
        output= model(x)
    
    return output[: int(rate * len(x))].view(-1)

def download(link):
    response= requests.get(link)
    if response.status_code == 200: 
        print('Cannot Install')
        filename= os.path.basename(link)
        with open('data/' + filename, 'wb') as f:
            f.write(response.content)

        return 'data/' + filename
    else: 
        return None

