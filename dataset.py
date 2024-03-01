import torchaudio
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from typing import List
import torchaudio.transforms as T

class VLSPDataset(Dataset):
    def __init__(
        self,
        processor:Wav2Vec2Processor,
        path='/kaggle/input/vin-big-data-vlsp-2020-100h',
        path_csv='/kaggle/input/datacsv/VLSP_under10s_train.csv',
    ):
        super().__init__()
        df = pd.read_csv(path_csv)
        df.path = path + os.sep + df.path
        self.walker = df.to_dict("records")
        self.processor =processor 
    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        path, trans = self.walker[idx]['path'],self.walker[idx]['text']
        trans=self.processor(text=trans.lower()).input_ids
        wave, sr = torchaudio.load(path)
        specs = self.processor(wave,sampling_rate=16000,output_tensor='pt').input_values[0][0]

        return  {'input_values':specs, 'labels':trans}
    
class SemiSupervisedDataset(Dataset):
    def __init__(
        self,
        processor:Wav2Vec2Processor,
        path_vlsp:str="/kaggle/input/vin-big-data-vlsp-2020-100h",
        path_vivos:str="/kaggle/input/vivos-vietnamese/vivos",
        path_commonv:str='/kaggle/input/commonvoice-vie/cv-corpus-15.0-2023-09-08',
        path_csv_vlsp :str= "/kaggle/input/datacsv/VLSP_under10s_train.csv",
        path_csv_vios:str='/kaggle/input/datacsv/vivos_upder10s_train.csv',
        path_csv_commonv:str='/kaggle/input/datacsv/commonvoice_train.csv',
        **kwargs,
    ):
        super().__init__()

       
        df_VLSP = pd.read_csv(path_csv_vlsp)
        df_VLSP.path = path_vlsp + os.sep + df_VLSP.path
        
   
        df_vivos=pd.read_csv(path_csv_vios)
        df_vivos.path=path_vivos+os.sep+df_vivos.path
        
        df_commonv=pd.read_csv(path_csv_commonv)
        df_commonv.path=path_commonv+os.sep+df_commonv.path
        
        walker_VLSP = df_VLSP.to_dict("records")
        walker_vivos=df_vivos['path'].to_list()
        walker_commonv=df_commonv['path'].to_list()
        
        self.walker=walker_vivos+walker_VLSP+walker_commonv
        random.shuffle(self.walker)
        self.processor=processor
    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]
        if type(item) == dict:
            return self.load_labeled_item(item)
        return self.load_unlabeled_item(item)

    def load_unlabeled_item(self, item):
        wave, sr = torchaudio.load(item)
        if sr >16000:
            resample_transform = T.Resample(orig_freq=sr, new_freq=16000)
            wave = resample_transform(wave)
        specs = self.processor(wave,sampling_rate=16000,output_tensor='pt').input_values[0][0]
        return  {'input_values':specs, 'labels':[]}       

    def load_labeled_item(self, item):
        path, trans = item['path'],item['text']
        trans=self.processor(text=trans.lower()).input_ids
        wave, sr = torchaudio.load(path)
        specs = self.processor(wave,sampling_rate=16000,output_tensor='pt').input_values[0][0]
        return  {'input_values':specs, 'labels':trans}

class DatasetValidated(Dataset):
    def __init__(
        self,
        processor:Wav2Vec2Processor,
        path='/kaggle/input/vivos-vietnamese/vivos',
        path_csv='/kaggle/input/datacsv/vivos_test.csv',

    ):
        super().__init__()
        df= pd.read_csv(path_csv)
        df.path=path+os.sep+df.path
        self.walker = df.to_dict("records")
        self.processor =processor 
    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        path, trans = self.walker[idx]['path'],self.walker[idx]['text']
        trans=self.processor(text=trans.lower()).input_ids
        wave, sr = torchaudio.load(path)
        if sr >16000:
            resample_transform = T.Resample(orig_freq=sr, new_freq=16000)
            wave = resample_transform(wave)
        specs = self.processor(wave,sampling_rate=16000,output_tensor='pt').input_values[0][0]
        return  {'input_values':specs, 'labels':trans}       