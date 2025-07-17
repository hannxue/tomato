import sys
import os
import torch
import torch.nn as nn
import transformers
import numpy as np
from peft import PeftModel
from datasets import load_from_disk
from transformers import LlamaConfig
from model.vamoe import Vmoe_llama3
from transformers import AutoTokenizer
from bert_score import BERTScorer
from model.vae_cluster import Vae_Cluster_Es
from model.config_llama3 import llama_config
from model.moe_layer_llama import MoeBlock_RS
from peft import LoraConfig, TaskType, get_peft_model
from utils.prompt_process import prompt_template,Prompt_Process
from utils.utils import save_gate_index, postprocessing
from pepler_utils.utils import bleu_score, rouge_score
from distinct_n import distinct_n_sentence_level, distinct_n_corpus_level
# ---------------------------- 参数封装类 ----------------------------
# 用于保存一些必要的参数（维度/聚类数等）
class Args:
    def __init__(self, embedding_size,latent_dim,num_cluster):
        self.embedding_size = embedding_size   # 用户/物品嵌入维度
        self.latent_dim = latent_dim           # VAE 隐变量维度
        self.num_cluster = num_cluster         # 聚类数
        
'''
    yelp: user: 27147 item: 20266
    tripadvisor: user: 9765 item: 6280
    amazon: user: 7506 item: 7360
'''
# ---------------------------- 数据 & 模型路径设置 ----------------------------
### Inference Setting
n_user         = 7506# amazon 数据集的用户数
n_item         = 7360 # amazon 数据集的物品数
latent_dim     = 128
num_cluster    = 4
embedding_size = 768
vae_model_path = ''     # 预训练的 VAE 模型路径
tokenizer_path = 'meta-llama/Llama-3.1-8B-Instruct'# LLaMA 模型路径
llm_model_path = '' # LLaMA 模型的 LoRA checkpoint 路径
data_path      = '' # HuggingFace 格式的数据集路径
excel_path     = '' # Excel 输出路径
txt_path       = '' # 评估指标保存路径
# 构建 Args 对象
args = Args(embedding_size = embedding_size, latent_dim = latent_dim, num_cluster = num_cluster)
config = llama_config  # LLaMA 的 config 已在外部定义好（模型结构信息）

# ---------------------------- 加载 VAE 模型 ----------------------------
vae_clu = Vae_Cluster_Es(n_user = n_user, n_item = n_item, args = args)
vae_clu.load_state_dict(torch.load(vae_model_path)) # 加载 VAE 预训练权重

# 提取用户和物品嵌入（供解释模块使用）
user_embeds = vae_clu.encoder.user_embeddings
item_embeds = vae_clu.encoder.item_embeddings
# ---------------------------- 加载分词器 & 数据 ----------------------------
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token


data = load_from_disk(data_path).select(range(10000))# 只取前 10000 条数据加速测试
# ---------------------------- 获取聚类索引 ----------------------------
test_cluster_index = save_gate_index(data, vae_clu.cuda())# 对每条样本生成聚类 ID（用于路由）
# ---------------------------- 构建解释生成模型（含 MoE 门控） ----------------------------
vmoe_llama3 = Vmoe_llama3(config = config,tokenizer = tokenizer,gate_index_list = test_cluster_index, user_embed = user_embeds, item_embed = item_embeds, use_lora = False)
lora_checkpoint = llm_model_path
# ---------------------------- 加载 LoRA 微调模型 ----------------------------
model = PeftModel.from_pretrained(vmoe_llama3, model_id = lora_checkpoint)

model = model.cuda()
# ---------------------------- 开始生成 & 记录 ----------------------------
import pandas as pd
from tqdm import tqdm
df = pd.DataFrame(columns=['userID','itemID','feature','original_text','generate_text', 'inference_time'])
count = 0
# init
# 初始化 GPU 计时器
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
total_inference_time = 0.0
# ---------------------------- 遍历数据逐条生成 ----------------------------
for d in data:
    user = torch.tensor(d['user']).unsqueeze(0)
    item = torch.tensor(d['item']).unsqueeze(0)
    input_ids = torch.tensor(d['input_ids'][:62]).unsqueeze(0).cuda()
    text = d['text']

    start_event.record()
    
    out = model.generate(user, item, input_ids)

    end_event.record()

    torch.cuda.synchronize()

    inference_time = start_event.elapsed_time(end_event)  


    total_inference_time += inference_time
    
    inference_time = start_event.elapsed_time(end_event)  
    generate_text = tokenizer.decode(out[0], skip_special_tokens=True)
    df = pd.concat([df,pd.DataFrame({'userID':d['user'],'itemID':d['item'],'feature':d['feature'],'original_text':text,'generate_text':generate_text, 'inference_time': inference_time},index=[0])], ignore_index=True)
    count = count + 1
    print(f'process {count}|{len(data["user"])}')
# ---------------------------- 输出平均推理时间 ----------------------------
average_inference_time = total_inference_time / len(data)
print(f'average_inference_time: {average_inference_time:.2f} ms')

df = df.to_excel(excel_path)
print('Excel already is saved {}'.format(excel_path))
    
### Evaluation Metric
# ---------------------------- 开始评估 BLEU & ROUGE ----------------------------
dfm = pd.read_excel(excel_path)
dfm = dfm.dropna() 
dfm = dfm[dfm['generate_text'].str.len() >= 2]
print(dfm.columns)

# 准备评估数据
# # Compute the Metric y_pred: [ [] , [] ]
y_true = []
y_pred = []
print(len(dfm))
for index, row in dfm.iterrows():
    
    original_text = [postprocessing(row['original_text'])]
    generate_text = [postprocessing(row['generate_text'])]
    
    # original_text = [row['original_text']]
    # generate_text = [row['generate_text']]
    
    original_text = original_text[0].split()
    generate_text = generate_text[0].split()
    
    y_true.append(original_text)
    y_pred.append(generate_text)
    y_pred_distinct.append(row['original_text'].split())
    y_pred_distinct.append(row['generate_text'].split())
    
# ---------------------------- 计算指标 ----------------------------
BLEU1 = bleu_score(y_true, y_pred, n_gram=1, smooth=True)
print('BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(y_true, y_pred, n_gram=4, smooth=True)
print('BLEU-4 {:7.4f}'.format(BLEU4))

text_test = [' '.join(tokens) for tokens in y_true]
text_predict = [' '.join(tokens) for tokens in y_pred]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary



# 保存评估结果\with open(txt_path, 'w') as f:
with open(txt_path, 'w') as f:
    f.write('BLEU-1 {:7.4f}\n'.format(BLEU1))
    f.write('BLEU-4 {:7.4f}\n'.format(BLEU4))
    for (k, v) in ROUGE.items():
        f.write('{} {:7.4f}\n'.format(k, v))
    


'''
bleu input: [ ['the', 'staff'],
            ['the', 'hotel', 'is', 'a']]
rouge input: ['the staff are very friendly and helpful',
            'the hotel is a great place to stay']
'''

    
