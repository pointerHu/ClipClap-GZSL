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
# from ClipClap-GZSL.WavCaps.retrieval.models.ase_model import ASE
import os, sys
# sys.path.append(os.getcwd())
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from WavCaps.retrieval.models.ase_model import ASE
# from /home/wh/.ssh/ClipClap-GZSL/ClipClap-GZSL/WavCaps/retrieval/models/ase_model.py
from ruamel import yaml
from datetime import datetime


# 从CSV文件中加载UCF类别描述
df2 = pd.read_csv('/home/wh/PycharmProjects/KDA/avgzsl_benchmark_datasets/UCF/class-split/UCF.csv')

# 加载动作名称的最大补充描述
# ucf_classes = df2['name'].tolist()
descriptions = {
    "description_1": df2['description_1'].tolist(),
    "description_2": df2['description_2'].tolist(),
    "description_3": df2['description_3'].tolist(),
}

# 传入零样本分类器
# zeroshot_weights = zeroshot_classifier_with_descriptions(ucf_classes, ucf_templates, descriptions, device)


# 根据提示模板使用CLIP对类别标签进行嵌入
def zeroshot_classifier(classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = {}
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class根据模板生成具体文本

            texts = clip.tokenize(texts).to(device) #tokenize 将文本转换为模型可处理的输入
            class_embeddings = model.encode_text(texts) #embed with text encoder 使用 CLIP 模型生成文本嵌入
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # 每个嵌入向量归一化
            class_embedding = class_embeddings.mean(dim=0)# 对模板生成的所有嵌入取平均值
            class_embedding /= class_embedding.norm()
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)
        # zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


# 在类别文本模板后添加补充描述
def zeroshot_classifier_with_descriptions(classnames, templates, descriptions, device):

    with torch.no_grad():
        zeroshot_weights = {}

        # 遍历每个类别的动作名称
        for idx, classname in enumerate(tqdm(classnames)):
            texts = []

            # 处理每个模板，将补充描述添加到动作文本中
            for template in templates:
                # 获取对应类别的补充描述
                description_1 = descriptions["description_1"][idx]
                description_2 = descriptions["description_2"][idx]
                description_3 = descriptions["description_3"][idx]

                # 拼接描述到模板后面
                text_with_descriptions = f"{template.format(classname)} {description_1} {description_2} {description_3}"
                texts.append(text_with_descriptions)

            # Tokenize模板+描述
            tokens = clip.tokenize(texts).to(device)

            # 生成嵌入向量
            class_embeddings = model.encode_text(tokens)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            # 对所有模板的嵌入取均值归一化，生成该类别的权重
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            # 保存到字典
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)

    return zeroshot_weights

# 将类别名称、模板和描述组合在一起时，生成的文本太长，超过了CLIP模型的上下文长度限制（77个token）
def zeroshot_classifier_with_descriptions2(classnames, templates, descriptions, device):
    with torch.no_grad():
        zeroshot_weights = {}

        for idx, classname in enumerate(tqdm(classnames)):
            all_embeddings = []

            # 处理每个模板
            for template in templates:
                base_text = template.format(classname)

                # 方案1: 只使用第一个描述，避免文本过长
                # description_1 = descriptions["description_1"][idx]
                # text = f"{base_text} {description_1}"
                #
                # # 确保文本不会太长
                # if len(text.split()) > 50:  # 粗略估计token数量
                #     text = " ".join(text.split()[:50])
                #
                # tokens = clip.tokenize([text]).to(device)
                # embedding = model.encode_text(tokens)
                # embedding /= embedding.norm(dim=-1, keepdim=True)
                # all_embeddings.append(embedding)

                # 方案2: 如果需要使用所有描述，可以分别处理每个描述
                for desc_key in ["description_1", "description_2", "description_3"]:
                    desc = descriptions[desc_key][idx]
                    text = f"{base_text} {desc}"
                    # if len(text.split()) > 50:
                    #     text = " ".join(text.split()[:50])
                    tokens = clip.tokenize(text, truncate=True).to(device)
                    embedding = model.encode_text(tokens)
                    embedding /= embedding.norm(dim=-1, keepdim=True)
                    all_embeddings.append(embedding)

            # 合并所有嵌入
            all_embeddings = torch.cat(all_embeddings, dim=0)
            class_embedding = all_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)

    return zeroshot_weights

#KDA方式实现
def generate_zeroshot_weights(classnames):
    """
    生成零样本分类的权重字典，键为类名，值为嵌入向量

    Args:
        attributes_ids: 需要处理的类别ID列表

    Returns:
        zeroshot_weights: 字典，键为类名，值为嵌入向量
    """
    zeroshot_weights = {}
    act_name = df2["name"].tolist()
    act_1 = df2["description_1"].tolist()
    act_2 = df2["description_2"].tolist()
    act_3 = df2["description_3"].tolist()

    # 确保索引在有效范围内
    valid_ids = [i for i in attributes_ids if i < len(act_name)]

    for class_id in tqdm(valid_ids):
        class_name = act_name[class_id]

        # 获取该类别的所有描述
        descriptions = [
            act_name[class_id],  # 类名本身
            act_1[class_id],  # 描述1
            act_2[class_id],  # 描述2
            act_3[class_id]  # 描述3
        ]

        # 使用CLIP模型处理所有描述
        with torch.no_grad():
            # 对每个描述进行编码
            text_embeds_list = []
            for desc in descriptions:
                # 确保描述不为空
                if not isinstance(desc, str) or not desc.strip():
                    continue

                # 使用CLIP tokenizer和模型处理文本
                inputs = clip.tokenize([desc], padding=True, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # 获取文本嵌入
                text_embed = clip(**inputs).text_embeds

                # 归一化嵌入向量
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embeds_list.append(text_embed)

            # 如果有有效的描述
            if text_embeds_list:
                # 堆叠所有嵌入并取平均值
                all_embeds = torch.cat(text_embeds_list, dim=0)
                class_embedding = all_embeds.mean(dim=0)

                # 再次归一化
                class_embedding = class_embedding / class_embedding.norm()

                # 存储到字典中
                zeroshot_weights[class_name] = class_embedding.cpu().detach().numpy().astype(np.float32)

    return zeroshot_weights

# files = pd.read_csv("./avgzsl_benchmark_datasets/UCF/class-split/UCF.csv")  # 路径错误
# act_name = files["name"].tolist()
# act_1 = files["description_1"].tolist()
# act_2 = files["description_2"].tolist()
# act_3 = files["description_3"].tolist()

# self.act_name = [f'A sound event video of "{action}".' for action in self.act_name]
# act_name = [f'{action}' for action in act_name]





def zeroshot_classifier_descriptions_non_templates(classnames, templates, descriptions, device):
    with torch.no_grad():
        zeroshot_weights = {}
        texts = []
        for idx, classname in enumerate(tqdm(classnames)):
            description_1 = descriptions["description_1"][idx]
            text = f"{description_1}"
            texts.append(text)

        # 合并所有嵌入
        texts = clip.tokenize(texts).to(device)  # tokenize 将文本转换为模型可处理的输入
        class_embeddings = model.encode_text(texts)  # embed with text encoder 使用 CLIP 模型生成文本嵌入
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # 每个嵌入向量归一化
        class_embedding = class_embeddings.mean(dim=0)  # 对模板生成的所有嵌入取平均值
        class_embedding /= class_embedding.norm()
        zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)
    return zeroshot_weights

# df = pd.read_csv('/home/aoq234/thesis/ClipClap-GZSL/avgzsl_benchmark_non_averaged_datasets/UCF/class-split/ucf_clip_class_names.csv')
df = pd.read_csv('/home/wh/.ssh/ClipClap-GZSL/ClipClap-GZSL/avgzsl_benchmark_non_averaged_datasets/UCF/class-split/ucf_clip_class_names.csv')
ucf_classes = df['clip_class_name'].tolist()

ucf_templates = [
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





# zeroshot_weights = zeroshot_classifier(ucf_classes, ucf_templates, device)
# 将对类别的文本描述加入到文本嵌入当中
# zeroshot_weights = zeroshot_classifier_with_descriptions(ucf_classes, ucf_templates, descriptions, device)
zeroshot_weights = zeroshot_classifier_with_descriptions2(ucf_classes, ucf_templates, descriptions, device)
# zeroshot_weights = zeroshot_classifier_descriptions_non_templates(ucf_classes, ucf_templates, descriptions, device)
zeroshot_weights = generate_zeroshot_weights(ucf_classes)

print(zeroshot_weights.keys())
# data_root_path = '/home/aoq234/akata-shared/aoq234/avzsl/clip_original/avgzsl_benchmark_datasets/UCF/features/cls_features_non_averaged'
data_root_path = '/home/wh/clip_original/avgzsl_benchmark_datasets/UCF/features/cls_features_non_averaged'
data_path = os.path.join(data_root_path, 'text')

if not(os.path.exists(data_path)):
    os.makedirs(data_path)
# filename = os.path.join(data_path, 'word_embeddings_ucf_normed.npy')
word_embedding = 'word_embeddings_ucf_description_normed'
file_extension = '.npy'
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
new_filename = f"{word_embedding}_{current_time}{file_extension}"
filename = os.path.join(data_path, new_filename)
# filename = os.path.join(data_path, 'word_embeddings_ucf_description_normed.npy')


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

# 在类别文本模板后添加补充描述
def wavcaps_zeroshot_classifier_with_descriptions(classnames, templates, descriptions, device):

    with torch.no_grad():
        zeroshot_weights = {}

        # 遍历每个类别的动作名称
        for idx, classname in enumerate(tqdm(classnames)):
            texts = []

            # 处理每个模板，将补充描述添加到动作文本中
            for template in templates:
                # 获取对应类别的补充描述
                description_1 = descriptions["description_1"][idx]
                description_2 = descriptions["description_2"][idx]
                description_3 = descriptions["description_3"][idx]

                # 拼接描述到模板后面
                text_with_descriptions = f"{template.format(classname)} {description_1} {description_2} {description_3}"
                texts.append(text_with_descriptions)

            class_embeddings = wavcaps_model.encode_text(texts)  # embed with text encoder
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)
    return zeroshot_weights


def wavcaps_zeroshot_classifier_with_descriptions2(classnames, templates, descriptions, device):
    with torch.no_grad():
        zeroshot_weights = {}

        for idx, classname in enumerate(tqdm(classnames)):
            texts = []

            for template in templates:
                base_text = template.format(classname)

                # 只使用第一个描述
                # description_1 = descriptions["description_1"][idx]
                # text = f"{base_text} {description_1}"
                #
                # # 确保文本不会太长
                # if len(text.split()) > 50:
                #     text = " ".join(text.split()[:50])
                # embedding = wavcaps_model.encode_text([text])
                # embedding /= embedding.norm(dim=-1, keepdim=True)
                # texts.append(embedding)
                for desc_key in ["description_1", "description_2", "description_3"]:
                    desc = descriptions[desc_key][idx]
                    text = f"{base_text} {desc}"
                    # if len(text.split()) > 50:
                    #     text = " ".join(text.split()[:50])

                    embedding = wavcaps_model.encode_text([text])
                    embedding /= embedding.norm(dim=-1, keepdim=True)
                    texts.append(embedding)

            class_embeddings = torch.cat(texts, dim=0)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)

    return zeroshot_weights




ucf_audio_templates = [
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
with open("/home/wh/.ssh/ClipClap-GZSL/ClipClap-GZSL/WavCaps/retrieval/settings/inference.yaml", "r") as f:
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

# wavecaps_zeroshot_weights = wavcaps_zeroshot_classifier(ucf_classes, ucf_audio_templates, device)
wavecaps_zeroshot_weights = wavcaps_zeroshot_classifier_with_descriptions(ucf_classes, ucf_audio_templates,descriptions, device)
# wavecaps_zeroshot_weights = wavcaps_zeroshot_classifier_with_descriptions2(ucf_classes, ucf_audio_templates,descriptions, device)


print(wavecaps_zeroshot_weights.keys())
data_path = os.path.join(data_root_path, 'text')

if not(os.path.exists(data_path)):
    os.makedirs(data_path)
# filename = os.path.join(data_path, 'wavcaps_word_embeddings_ucf_normed.npy')
# filename = os.path.join(data_path, 'wavcaps_word_embeddings_ucf_description_normed.npy')
wavcaps_word_embedding = 'wavcaps_word_embeddings_ucf_description_normed'
# file_extension = '.npy'
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
new_filename = f"{wavcaps_word_embedding}_{current_time}{file_extension}"
filename = os.path.join(data_path, new_filename)

np.save(filename, wavecaps_zeroshot_weights)


# python clip_embeddings_extraction/get_clip_embeddings_ucf.py
