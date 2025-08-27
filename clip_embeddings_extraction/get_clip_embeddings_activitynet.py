import clip
from pathlib import Path
import os
import sys
import torch
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from prompt_toolkit import prompt
import pandas as pd
from tqdm import tqdm

# 解决pycharm有时索引不到文件
import os, sys
# sys.path.append(os.getcwd())
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from WavCaps.retrieval.models.ase_model import ASE
from ruamel import yaml
import argparse
from src.args import str_to_bool
from datetime import datetime

def zeroshot_classifier(classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = {}
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)
        # zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

# 从CSV文件中加载UCF类别描述
df2 = pd.read_csv('../avgzsl_benchmark_non_averaged_datasets/ActivityNet/class-split/ActivityNet.csv')
descriptions = {
    "description_1": df2['description_1'].tolist(),
    "description_2": df2['description_2'].tolist(),
    "description_3": df2['description_3'].tolist(),
}
# 将类别名称、模板和描述组合在一起时，生成的文本太长，超过了CLIP模型的上下文长度限制（77个token）
def zeroshot_classifier_with_descriptions(classnames, templates, descriptions, device):
    with torch.no_grad():
        zeroshot_weights = {}

        for idx, classname in enumerate(tqdm(classnames)):
            all_embeddings = []

            # 处理每个模板
            for template in templates:
                base_text = template.format(classname)
                # 方案2: 如果需要使用所有描述，可以分别处理每个描述
                for desc_key in ["description_1", "description_2", "description_3"]:
                    desc = descriptions[desc_key][idx]
                    text = f"{base_text} {desc}"
                    tokens = clip.tokenize(text).to(device)
                    embedding = model.encode_text(tokens)
                    embedding /= embedding.norm(dim=-1, keepdim=True)
                    all_embeddings.append(embedding)

            # 合并所有嵌入
            all_embeddings = torch.cat(all_embeddings, dim=0)
            class_embedding = all_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)

    return zeroshot_weights


# df = pd.read_csv('/home/aoq234/akata-shared/aoq234/avzsl/clip_original/avgzsl_benchmark_datasets/ActivityNet/class-split/activitynet_w2v_class_names.csv')
df = pd.read_csv('../avgzsl_benchmark_non_averaged_datasets/ActivityNet/class-split/activitynet_w2v_class_names.csv')
activitynet_classes = df['manual'].tolist()


activitynet_templates = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'a example of a person {}.',
    'a demonstration of a person {}.',
    'a photo of the person {}.',
    'a video of the person {}.',
    'a example of the person {}.',
    'a demonstration of the person {}.',
    'a photo of a person using {}.',
    'a video of a person using {}.',
    'a example of a person using {}.',
    'a demonstration of a person using {}.',
    'a photo of the person using {}.',
    'a video of the person using {}.',
    'a example of the person using {}.',
    'a demonstration of the person using {}.',
    'a photo of a person doing {}.',
    'a video of a person doing {}.',
    'a example of a person doing {}.',
    'a demonstration of a person doing {}.',
    'a photo of the person doing {}.',
    'a video of the person doing {}.',
    'a example of the person doing {}.',
    'a demonstration of the person doing {}.',
    'a photo of a person during {}.',
    'a video of a person during {}.',
    'a example of a person during {}.',
    'a demonstration of a person during {}.',
    'a photo of the person during {}.',
    'a video of the person during {}.',
    'a example of the person during {}.',
    'a demonstration of the person during {}.',
    'a photo of a person performing {}.',
    'a video of a person performing {}.',
    'a example of a person performing {}.',
    'a demonstration of a person performing {}.',
    'a photo of the person performing {}.',
    'a video of the person performing {}.',
    'a example of the person performing {}.',
    'a demonstration of the person performing {}.',
    'a photo of a person practicing {}.',
    'a video of a person practicing {}.',
    'a example of a person practicing {}.',
    'a demonstration of a person practicing {}.',
    'a photo of the person practicing {}.',
    'a video of the person practicing {}.',
    'a example of the person practicing {}.',
    'a demonstration of the person practicing {}.',
]



# device = 'cuda:3'
device = 'cuda:0'
model, preprocess = clip.load("ViT-B/32", device=device)



model = model.to(device)
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

model_version = 'clip_original'


# zeroshot_weights = zeroshot_classifier(activitynet_classes, activitynet_templates, device)
zeroshot_weights = zeroshot_classifier_with_descriptions(activitynet_classes, activitynet_templates, descriptions, device)


print(zeroshot_weights.keys())
# data_root_path = '/home/wh/clip_original/avgzsl_benchmark_datasets/UCF/features/cls_features_non_averaged'
data_root_path = f'/home/wh/{model_version}/avgzsl_benchmark_datasets/ActivityNet/features/cls_features_non_averaged'
data_path = os.path.join(data_root_path, 'text')

if not(os.path.exists(data_path)):
    os.makedirs(data_path)
# filename = os.path.join(data_path, 'word_embeddings_activitynet_normed.npy')

word_embedding = 'word_embeddings_activitynet_description_normed'
file_extension = '.npy'
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
new_filename = f"{word_embedding}_{current_time}{file_extension}"
filename = os.path.join(data_path, new_filename)

np.save(filename, zeroshot_weights)















def wavcaps_zeroshot_classifier(classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = {}
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class

            class_embeddings = wavcaps_model.encode_text(texts) #embed with text encoder
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)
    return zeroshot_weights

def wavcaps_zeroshot_classifier_with_descriptions(classnames, templates, descriptions, device):
    with torch.no_grad():
        zeroshot_weights = {}
        for idx, classname in enumerate(tqdm(classnames)):
            texts = []
            for template in templates:
                base_text = template.format(classname)
                for desc_key in ["description_1", "description_2", "description_3"]:
                    desc = descriptions[desc_key][idx]
                    text = f"{base_text} {desc}"
                    embedding = wavcaps_model.encode_text([text])
                    embedding /= embedding.norm(dim=-1, keepdim=True)
                    texts.append(embedding)
            class_embeddings = torch.cat(texts, dim=0)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)
    return zeroshot_weights





activitynet_audio_templates = [
    'a person {} can be heard.',
    'a example of a person {} can be heard.',
    'a demonstration of a person {} can be heard.',
    'the person {} can be heard.',
    'a example of the person {} can be heard.',
    'a demonstration of the person {} can be heard.',
    'a person using {} can be heard.',
    'a example of a person using {} can be heard.',
    'a demonstration of a person using {} can be heard.',
    'a example of the person using {} can be heard.',
    'a demonstration of the person using {} can be heard.',
    'a person doing {} can be heard.',
    'a example of a person doing {} can be heard.',
    'a demonstration of a person doing {} can be heard.',
    'a example of the person doing {} can be heard.',
    'a demonstration of the person doing {} can be heard.',
    'a example of a person during {} can be heard.',
    'a demonstration of a person during {} can be heard.',
    'a example of the person during {} can be heard.',
    'a demonstration of the person during {} can be heard.',
    'a person performing {} can be heard.',
    'a example of a person performing {} can be heard.',
    'a demonstration of a person performing {} can be heard.',
    'a example of the person performing {} can be heard.',
    'a demonstration of the person performing {} can be heard.',
    'a person practicing {} can be heard.',
    'a example of a person practicing {} can be heard.',
    'a demonstration of a person practicing {} can be heard.',
    'a example of the person practicing {} can be heard.',
    'a demonstration of the person practicing {} can be heard.'
]


# with open("/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/settings/inference.yaml", "r") as f:
with open("../WavCaps/retrieval/settings/inference.yaml", "r") as f:
    config = yaml.safe_load(f)
# device = 'cuda:3'
device = 'cuda:0'
wavcaps_model = ASE(config)
wavcaps_model.to(device)



# cp_path = '/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/pretrained_models/audio_encoders/HTSAT_BERT_zero_shot.pt'
cp_path = '/home/wh/.ssh/ClipClap-GZSL/ClipClap-GZSL/WavCaps/retrieval/pretrained_models/HTSAT_BERT_zero_shot.pt'
state_dict_key = 'model'
cp = torch.load(cp_path)
wavcaps_model.load_state_dict(cp[state_dict_key])
wavcaps_model.eval()
print("Model weights loaded from {}".format(cp_path))

# wavecaps_zeroshot_weights = wavcaps_zeroshot_classifier(activitynet_classes, activitynet_audio_templates, device)
wavecaps_zeroshot_weights = wavcaps_zeroshot_classifier_with_descriptions(activitynet_classes, activitynet_audio_templates, descriptions, device)


print(wavecaps_zeroshot_weights.keys())
data_path = os.path.join(data_root_path, 'text')

if not(os.path.exists(data_path)):
    os.makedirs(data_path)
# filename = os.path.join(data_path, 'wavcaps_word_embeddings_activitynet_normed.npy')

wavcaps_word_embedding = 'wavcaps_word_embeddings_activitynet_description_normed'
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
new_filename = f"{wavcaps_word_embedding}_{current_time}{file_extension}"
filename = os.path.join(data_path, new_filename)

np.save(filename, wavecaps_zeroshot_weights)


# python clip_embeddings_extraction/get_clip_embeddings_activitynet.py
