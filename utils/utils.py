'''
    utlls.py: tool class
'''
import os
import re#正则表达式库，用于文本匹配与替换。
import torch
from collections import Counter#统计容器中每个元素的出现次数。
from sklearn.manifold import TSNE#t-SNE 算法，用于把高维数据降到 2D 或 3D，可视化聚类结果。
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from typing import Optional#提供类型提示功能。
from transformers import PreTrainedTokenizerBase#HuggingFace 中 tokenizer 的基类
from datasets import Dataset as HFDataset#HuggingFace 的数据集格式，兼容 Trainer，支持数据处理、缓存、切分等。

from transformers.trainer_utils import EvalLoopOutput#Trainer 中评估输出的标准结构。
from transformers import Trainer#HuggingFace 的通用训练类，封装了训练、评估、保存、日志记录等流程。
from transformers.utils import logging#HuggingFace 提供的日志记录模块。
from torch.utils.data import SequentialSampler#顺序采样器，用于按顺序加载数据而不是打乱。

'''
    process_explain_data_fun: 
        args:
            examples: single data
        pad、tokenize、add bos eos token
'''
# ---------------------------------------------
# 将 PyTorch 数据集转换为 HuggingFace 数据集格式，方便使用 Trainer
# ---------------------------------------------
def TorchDataset2HuggingfaceDataset(torch_dataset, cache_dir = None):
    generator = lambda: (sample for sample in torch_dataset)  # 使用生成器封装原始 PyTorch 数据 
    return HFDataset.from_generator(generator, cache_dir=cache_dir)
# ---------------------------------------------
# 文本处理函数，用于单条样本。增加 BOS/EOS token，保留用户、物品信息
# ---------------------------------------------
def process_fun(examples):
    # examples['text'] = '<>'
    # 对输入的文本字段进行 tokenizer 编码（截断最大长度为 20）
    encode_inputs = tokenizer(examples['text'], max_length = 20, truncation = True)
    # 手动保留 user 和 item 字段，用于个性化推荐
    encode_inputs["user"] = examples["user"] # 添加用户信息
    encode_inputs["item"] = examples["item"] # 添加物品信息
    # encode_inputs["rating"] = examples["rating"]
    # 在 input_ids 前加上 BOS token
    for key, value in tokenizer(tokenizer.bos_token).items():
        encode_inputs[key] = value + encode_inputs[key]
    
    # 在 input_ids 后加上 EOS token
    for key, value in tokenizer(tokenizer.eos_token).items():
        encode_inputs[key] = encode_inputs[key] + value
        
    return encode_inputs

# ---------------------------------------------
# 类形式封装的解释生成数据预处理
# ---------------------------------------------
class Process_Explain_data:
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase], max_seq_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples):
        # 编码 explanation 字段为 input_ids 等 token
        model_inputs = self.tokenizer(examples["explanation"], 
                                      max_length=self.max_seq_length,
                                      truncation=True)
        # 附加结构化字段（用于生成模型做个性化解释）
        model_inputs["user"] = examples["user"]
        model_inputs["item"] = examples["item"]
        model_inputs["rating"] = examples["rating"]
        
        # add prefix and postfix key: input_ids 
        # 添加 BOS token 到前缀
        for key, value in self.tokenizer(self.tokenizer.bos_token).items():
            model_inputs[key] = value + model_inputs[key]
        
        # 添加 EOS token 到后缀
        for key, value in self.tokenizer(self.tokenizer.eos_token).items():
            model_inputs[key] = model_inputs[key] + value

        # until this step, the length of each example input_ids is not equal
        return model_inputs


# ---------------------------------------------
# 可视化潜在空间中的聚类分布，用 t-SNE 降维到 2D
# ---------------------------------------------
def plot_latent(vae_clu, data_loader, args, epoch):
    with torch.no_grad():# 禁用梯度，提高效率
        Z = [] # 存储编码器输出的 latent 向量
        Y = [] # 存储预测的类别索引
        vae_clu.eval() # 设置模型为评估模式
        
        for batch_index,(user, item, rating, _, _) in enumerate(data_loader):
            user = user.to('cuda')
            item = item.to('cuda')
            z1, _, _ = vae_clu.encoder(user, item)  # 获取用户-物品的 latent 表征
            y = vae_clu.predict(user,item)# 获取聚类类别索引
            Y.append(torch.tensor(y))
            Z.append(z1)
        # [batch, latent_dim]
        Z = torch.cat(Z, 0).detach().cpu().numpy() # 合并并转为 numpy
        Y = torch.cat(Y, 0).detach().cpu().numpy()
        index_counts = Counter(Y)
        # 打印每个聚类出现次数
        for index, count in index_counts.items():
            print(f"Cluster {index} appears {count} times.")
 
        print(f'🤡🤡🤡 Ploting Latent Space for {args.dataset}')
        num_samples_per_cluster = 300
        # # 每类随机选 300 个样本进行可视化
        indices = []
        for i in range(args.num_cluster):
            indices_in_cluster = np.where(Y == i)[0]
            selected_indices = np.random.choice(indices_in_cluster, num_samples_per_cluster, replace=True)
            indices.extend(selected_indices)

        selected_Z = Z[indices]
        selected_Y = Y[indices]


        tsne = TSNE(n_components=3, init='pca', random_state=42)  
        Z_tsne = tsne.fit_transform(selected_Z)
        # —— 投射到单位球面，便于 3D 球面直观展示
        norms = np.linalg.norm(Z_tsne, axis=1, keepdims=True) + 1e-12
        Z_sph = Z_tsne / norms                   # (N, 3)

        
        # ========== 绘图 ==========
        fig = plt.figure(figsize=(14, 14))
        ax  = fig.add_subplot(111, projection="3d")

        
        cmap_name = 'tab10' if args.num_cluster <= 10 else 'tab20'   # 自动选 10 色或 20 色
        cmap      = get_cmap(cmap_name)
        unique_clusters = np.unique(selected_Y)
        for i, k in enumerate(unique_clusters):
            idx =  selected_Y == k
            color = cmap(i % cmap.N)        # cmap.N = 10 或 20
            ax.scatter(Z_sph[idx, 0], Z_sph[idx, 1], Z_sph[idx, 2],
                   s=8, alpha=0.8, color=color, 
                   label=f'Cluster {k}')
        # —— 绘制单位球（wireframe）
        theta = np.linspace(0, 2 * np.pi, 100)
        phi   = np.linspace(0, np.pi,     100)
        THETA, PHI = np.meshgrid(theta, phi)
        X_sph = np.sin(PHI) * np.cos(THETA)
        Y_sph = np.sin(PHI) * np.sin(THETA)
        Z_sph0= np.cos(PHI)
        ax.plot_wireframe(X_sph, Y_sph, Z_sph0,
                      color="gray", linewidth=0.5, alpha=0.2)

        # —— 视角 & 美化
        ax.view_init(elev=20, azim=60)   # 旋转角度可自行调整
        ax.set_box_aspect([1, 1, 1])     # 保证球体不被拉伸
        ax.set_axis_off()                # 隐藏坐标轴
        ax.legend(loc="upper left", fontsize=10, frameon=False)

         # —— 保存

        plt.savefig(os.path.join(args.pretrain_weight_save,args.dataset, args.dataset + '_' + f'latent_vis_cluster_{args.num_cluster}_epoch_{epoch}.png'),dpi=300, bbox_inches="tight")
        plt.show()
        print(f'Plot Latent Space Done for {epoch}')



# ---------------------------------------------
# 给字典中某个键扩展值（值为 list）
# ---------------------------------------------
def dict_extend(dict, key, value):
    """
        extend the list value of key in dict
    """
    if key in dict and isinstance(value,list):
            dict[key].extend(value)
    else:
        dict[key] = value 
# ---------------------------------------------
# 判断结构中是否包含 Tensor（递归）
# ---------------------------------------------
def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    Credit: AllenNLP
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False

# ---------------------------------------------
# 自定义 Trainer，重写采样器，强制顺序加载数据（适用于推荐）
#推荐系统往往具有用户行为时序性或用户-物品交互对的结构化特点，打乱顺序会破坏这种时序或结构一致性
# --------------------------------------------
class RecTrainer(Trainer):
    def __init__(self, *args, save_lora = True, **kwargs):
        self.save_lora = save_lora# 控制是否保存 LoRA 微调结果
        super().__init__(*args, **kwargs)#然后调用父类初始化 Trainer（模型、数据集、训练参数等）。

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        
        return SequentialSampler(self.train_dataset)## 强制使用顺序采样器，避免打乱顺序


import torch
from torch.utils.data import DataLoader

# ---------------------------------------------
# 对完整数据进行聚类预测，生成 gate index（门控路由）列表
# 对输入的 HuggingFace 格式数据集中的所有用户-物品对，使用训练好的 VAE + GMM 模型进行聚类预测，得到每个样本所属的簇编号（即门控索引）
# ---------------------------------------------
def save_gate_index(hf_dataset, vae_clu, batch_size=1000):
    cluster_index_list = []# 初始化一个空列表，用于保存每个样本的聚类（簇）编号。
    #移除不必要的字段，只保留 user 和 item 两列，因为聚类预测只需要用户和物品的 ID。
    hf_dataset = hf_dataset.remove_columns(['labels','feature', 'input_ids', 'attention_mask','rating'])
    data_loader = DataLoader(hf_dataset, batch_size=batch_size, shuffle=False)#使用 PyTorch 的 DataLoader 封装数据，方便按批次进行推理（不打乱顺序）。

    print('Processing the gate index...')
    total_batches = len(data_loader)
    processed_batches = 0

    for batch in data_loader:
        users = torch.tensor(batch['user']).to(vae_clu.device)
        items = torch.tensor(batch['item']).to(vae_clu.device)

        indices = vae_clu.predict(users, items)#使用训练好的模型 vae_clu 的方法 predict_cluster_index()，对当前 batch 的每个样本预测它属于哪个簇（cluster index）
        cluster_index_list.extend(indices.tolist()) # 把当前 batch 的聚类编号结果追加进总列表中。
    
        processed_batches += 1
        if processed_batches % 1000 == 0:
            print(f'process {processed_batches} / {total_batches}')
    
    print(f'Save Gate Index List Length: {len(cluster_index_list)}')
    return cluster_index_list
# ---------------------------------------------
# 英文句子规范化处理，方便 tokenizer 分词
# ---------------------------------------------
def postprocessing(string):
    '''
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string