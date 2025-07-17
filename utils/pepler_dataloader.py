'''
    code ref: https://github.com/lileipisces/PEPLER/blob/master/utils.py
'''
import os
import re
import math
import torch
import random
import pickle
from torch.utils.data import Dataset,DataLoader

#将用户 / 物品字符串 ID 压缩为 0-N 的整数索引，可直接作为嵌入表下标；节省显存、加速 lookup。
class EntityDictionary:
    def __init__(self):
        self.idx2entity = [] # 用 list 保存 “索引 → 实体” 的正向映射, 正向查询：已知整数索引，O(1) 取出实体字符串（idx2entity[i]）
        self.entity2idx = {} # 用 dict 保存 “实体 → 索引” 的反向映射, 反向查询：已知实体字符串，O(1) 找到其连续索引（entity2idx[s]）

    def add_entity(self, e):
         # 如果该实体 e 之前没出现过，就分配一个新的顺序索引
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity) # 反向表：e → 新索引
            self.idx2entity.append(e)                 # 正向表：索引 → e

    def __len__(self):
        return len(self.idx2entity)# 让 len(obj) 直接返回实体个数


class DataLoader_Rs:
    # ---------------------------------------------------------------------
    # 该类负责三件事：
    #   1. 读取 pickle 格式的评价列表（reviews），建立 user/item → idx 字典
    #   2. 将原始 review 划分为 train / valid / test 三个列表
    #   3. 收集每个用户、物品出现过的 feature，用于冷启动或解释
    # WHY: 把「数据读取 + 划分 + 统计」集中封装，主脚本只需一行即可拿到
    #      train/valid/test + 字典，符合单一职责原则。
    # ---------------------------------------------------------------------
    def __init__(self, data_path, index_dir, tokenizer, seq_len):
        # user_dict / item_dict: 负责把字符串 ID 映射到连续整数索引
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        # 初始化评分的最大值和最小值，后续可做归一化或分桶(比如按评分区间分为“低评分”、“中等评分”、“高评分”三类。),
        self.max_rating = float('-inf')
        self.min_rating = float('inf')

        # 先遍历一遍原始文件，填充字典并统计评分范围
        self.initialize(data_path)

        
        self.feature_set = set()#初始化一个空的集合，用于存储所有出现过的特征或标签字段。

        # tokenizer / seq_len: 序列化文本时需要截断到固定长度
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # 真正加载并切分数据；同时返回 user→feature / item→feature 索引
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(data_path, index_dir)
    

    # ---------------------------------------------------------------------
    # 一次性扫描整个评分文件，作用：
    #   * 建立 user_dict / item_dict 的映射
    #   * 提前得到 max_rating / min_rating，方便后续归一化
    # WHY: 只读磁盘一次即可拿到全局统计，避免反复 I/O。
    # ---------------------------------------------------------------------
    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))# 读取序列化文件
        for review in reviews:
            self.user_dict.add_entity(review['user'])##将用户 ID 和物品 ID 加入字典（构建 user_dict 和 item_dict 映射表）
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:#动态更新评分上下界：即 self.max_rating 和 self.min_rating
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating
    

    # ---------------------------------------------------------------------
    # 将 reviews 划分为 train / valid / test，并补充辅助字典：
    #   user2feature: {user_idx: [feature1, feature2, ...]}
    #   item2feature: {item_idx: [feature1, feature2, ...]}
    # WHY:
    #   * user2feature / item2feature 在解释生成阶段可作为先验信息
    #   * 读取 index 文件而不是随机划分，可保证与原论文/实验对齐
    # ---------------------------------------------------------------------
    def load_data(self, data_path, index_dir):#data_path：原始数据（通常是 .pkl 格式）,
        data = []
        reviews = pickle.load(open(data_path, 'rb'))#每条 review 是一个字典，包含用户、物品、评分、模板等字段。
        # 1️⃣ 先将所有 review 转为数值化字典并累计 feature_set
        for review in reviews:
            (fea, adj, tem, sco) = review['template']#从 template 中解析出：fea: 特征（如“服务”、“环境”）;adj: 描述词（如“很好”）;tem: 模板文本;sco: 分数
            data.append({'user': self.user_dict.entity2idx[review['user']],#将用户和物品的原始 ID 映射为整数索引（用字典）,构造新的格式统一的样本，加入 data 列表
                         'item': self.item_dict.entity2idx[review['item']],#比如用户 "u1" → 0，物品 "i3" → 2
                         'rating': review['rating'],
                         'text': tem, # 填好后面 prompt 会用到
                         'feature': fea}) # eg. "environment"
            self.feature_set.add(fea)#收集所有出现过的特征，形成一个集合
        
        # 2️⃣ 根据 index 文件读取三折划分的行号
        train_index, valid_index, test_index = self.load_index(index_dir)
        
        # 3️⃣ 生成具体数据列表 + 统计 user/item 对应的 feature
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]#从 data 中取出指定索引的样本，加到 train 列表中
            train.append(review)
            u = review['user']#分别获取样本中的用户 ID、物品 ID、对应特征
            i = review['item']
            f = review['feature']
            # 用户的 feature 多值累积
            if u in user2feature:#记录这个用户在训练集中说过的所有特征
                user2feature[u].append(f)#user2feature[0] = ['服务']
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)#item2feature[2] = ['服务']
            else:
                item2feature[i] = [f]

        # 验证集 / 测试集只需列表，无需统计 feature
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature


    # ---------------------------------------------------------------------
    # 读取事先生成好的三折索引：
    #   train.index / validation.index / test.index
    # WHY: 固定划分保证可复现；不同实验共享同一 splits。
    # ---------------------------------------------------------------------
    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:#这个文件里保存的是训练数据的样本编号 0 1 4 6 8 10
            train_index = [int(x) for x in f.readline().split(' ')]#'0 1 4 6 8 10'.split(' ')  → ['0', '1', '4', '6', '8', '10']
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index

class Batchify:
    def __init__(self, data, batch_size=2, shuffle=False):
        u, i, r, t, self.feature = [], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            self.feature.append(x['feature'])


        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]

        return user, item, rating


# ---------------------------------------------
# 自定义 PyTorch 数据集类：用于推荐系统任务中的评分数据封装
# 输入数据是一个列表，其中每个元素是一个包含用户、物品、评分、文本和特征的字典
# ---------------------------------------------
class Dataset_Rs_Pytorch(Dataset):
    '''
        corpus_data: list 类型，每个元素是一个字典，形如：
        args: corpus_data: list of dict : [{'user': 3,# 用户 ID（已编码为整数）
                    'item': 0,                        # 物品 ID（已编码为整数）
                    'rating': 5,                      # 用户对该物品的评分
                    'text': ' can absolutely recommend this.',  # 用户的评论文本（解释用）
                    'feature': 'this'},                         # 样本对应的属性特征（如服务/位置）
                    {'user': 6154,
                    'item': 1769,
                    'rating': 5,
                    'text': 'atrium view with great artwork and flowers is worth the visit',
                    'feature': 'artwork'},]
    '''
    def __init__(self, corpus_data):
        self.corpus_data = corpus_data# 初始化，保存输入数据
    def __len__(self):
        return len(self.corpus_data)# 返回数据集大小（用于 len(dataset)）
    def __getitem__(self,index):
        '''
        支持通过索引访问数据集中的样本，如 dataset[10]
        返回一个五元组：(user, item, rating, text, feature)
        '''
        rs_data = self.corpus_data[index]#  取出指定索引的数据项
        user = torch.tensor(rs_data['user'], dtype=torch.int64).contiguous() # 用户 ID，转为 PyTorch tensor
        item = torch.tensor(rs_data['item'], dtype=torch.int64).contiguous() # 物品 ID
        rating = torch.tensor(rs_data['rating'], dtype=torch.int64).contiguous() # 评分（整数）
        text = rs_data['text']# 原始文本，不转 tensor，因为 tokenizer 之后才处理
        feature = rs_data['feature']  # 对应特征词，例如 "price", "service"
        return user, item, rating , text, feature
    