import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

class VITSInferer():
    def __init__(self,_hparam_path,_pth_path) -> None:
        self.hparam_path=_hparam_path
        self.pth_path=_pth_path
        self.hps=utils.get_hparams_from_file(self.hparam_path)
        self.net=self.load_model()
    
    # 文本预处理
    def get_norm_text(self,text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        
        return text_norm

    # 配置加载和生成器初始化
    def load_model(self):
        net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).cuda()
        _ = net_g.eval()
        # config path here
        _ = utils.load_checkpoint(pth_path, net_g, None)

        return net_g
    
    def generate(self,text,output_path="./output.wav"):
        norm_text=self.get_norm_text(text)
        with torch.no_grad():
            x_text = norm_text.cuda().unsqueeze(0)
            x_text_lengths = torch.LongTensor([x_text.size(0)]).cuda()
            audio = self.net.infer(x_text, x_text_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            # 采样率要注意
            write(output_path, 22050, audio)


if __name__=="__main__":
    hparam_path="./configs/ljs_base.json"
    pth_path="./logs/ljs_base/G_320000.pth"
    g=VITSInferer(hparam_path,pth_path)
    g.generate("VITS is Awesome!")