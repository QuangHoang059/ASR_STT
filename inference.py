from pyctcdecode import build_ctcdecoder
from multiprocessing import get_context,get_start_method
import torch
from datasets  import load_metric
from tqdm import  tqdm
import torchaudio.transforms as T

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

class Inference:
    def __init__(self,model,processor):
        self.model=model
        self.processor=processor
        if model.config.vocab_size==len(processor.tokenizer.get_vocab()):
            labels=list(dict(sorted(processor.tokenizer.get_vocab().items(), key=lambda item: item[1])))
        else:
            labels=list(dict(sorted(processor.tokenizer.vocab.items(), key=lambda item: item[1])))
        self.decoder=build_ctcdecoder(labels=labels,
            kenlm_model_path='LM/vi_lm_4grams.bin',
            alpha = 0.5,
            beta= 1.5,
            unk_score_offset=-10.0,
            lm_score_boundary=True,)
           
        self.wer_metric = load_metric("wer")
    def withtLM(self,logits,beam_width=100):
        pool=get_context('spawn').Pool(2)
        pred_strs=self.decoder.decode_batch(pool,logits.cpu().detach().numpy(), beam_width=beam_width)
        return pred_strs
    def withoutLM(self,logits):
        pred_ids = torch.argmax(logits, axis=-1)
        pred_str = self.processor.batch_decode(pred_ids)
        return pred_str
    def test_wer(self,test_dataloader,beam_width=None):
        wer_metric = load_metric("wer")
        running_wers=[]
        for batch in tqdm(test_dataloader,desc="Processing"):
            pred_str=self.bach_predict(batch['input_values'],beam_width)
            # we do not want to group tokens when computing the metrics
            batch['labels'][batch['labels']== -100] = self.processor.tokenizer.pad_token_id
            label_str =self.processor.batch_decode(batch['labels'], group_tokens=False)
            wer = wer_metric.compute(predictions=pred_str, references=label_str)
            running_wers.append(wer)
        return running_wers
    def bach_predict(self,batch,beam_width=None):
        with torch.no_grad():
            pred=self.model(batch.to(device))
        pred_logits = pred.logits
        if beam_width:
            pred_str=self.withtLM(pred_logits,beam_width)
        else:
            pred_str=self.withoutLM(pred_logits)
        return pred_str
    def predict(self,wave,sr,beam_width=None):
        if sr >16000:
            resample_transform = T.Resample(orig_freq=sr, new_freq=16000)
            wave = resample_transform(wave)
        specs = self.processor(wave,sampling_rate=16000,output_tensor='pt').input_values[0][0]

        with torch.no_grad():
            pred= self.model(torch.tensor([specs]).to(device))
        pred_logits = pred.logits
        if beam_width:
            pred_str=self.decoder.decode(pred_logits.cpu().detach().numpy()[0],beam_width=beam_width)
        else: 
            pred_str= self.withoutLM(pred_logits)
        return pred_str