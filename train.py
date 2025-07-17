import sys
import os
import argparse
import random
import numpy as np
import torch 
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()
from model.vae_cluster import Vae_Cluster_Es
from transformers import AutoTokenizer#AutoTokenizer：自动加载与你指定的预训练模型（如LLaMA）配套的分词器。用于将文本转为 token ID 输入模型。
from rich.console import Console#Console：用于在终端中打印带颜色/格式的内容，增强可读性（如日志、状态提示）
from tqdm import tqdm#tqdm：为循环添加进度条显示，提升训练可视化体验。
from torch.optim.lr_scheduler import StepLR#StepLR：PyTorch 的学习率调度器，按固定 epoch 步长减少学习率（防止过拟合或训练停滞）。
from torch.utils.data import DataLoader,Dataset
from utils.pepler_dataloader import Dataset_Rs_Pytorch,DataLoader_Rs
from collections import Counter
from transformers import AutoTokenizer,Trainer,TrainingArguments,DataCollatorWithPadding,EarlyStoppingCallback,DataCollatorForSeq2Seq#AutoTokenizer：自动加载与你指定的预训练模型（如LLaMA）配套的分词器。用于将文本转为 token ID 输入模型。
# HF 官方高阶训练循环（用在 RecTrainer 基类）# 所有超参集中管理 + 自动生成 --help # DataCollatorWithPadding, 对齐不同长度输入，常用于分类任务  #DataCollatorForSeq2Seq,  生成式任务 padding + label shift
from datasets import load_from_disk
from utils.utils import TorchDataset2HuggingfaceDataset,plot_latent,RecTrainer,save_gate_index#将 PyTorch 格式数据转换为 HuggingFace 数据集格式。可视化 latent space（VAE 的 z 空间）聚类结构，输出图片。自定义的 Trainer 类，封装训练逻辑（兼容 LoRA / 多任务等）。保存聚类分配的 gate 索引（用于解释生成模型的门控机制）。
from utils.prompt_process import Prompt_Process#数据预处理函数，将用户/物品/评分信息转换为自然语言解释 prompt，例如：“User X likes item Y with 4 stars because…”
from peft import LoraConfig, TaskType, get_peft_model#LoraConfig一种参数高效微调技术，只引入少量可训练参数，避免微调整个大模型。它通过插入低秩矩阵近似到原始权重中，显著减少计算和存储成本。
                                                     #TaskType这是一个枚举类，用于告诉 PEFT 当前的模型任务是什么。
                                                     #get_peft_model将原始的预训练模型（如 LLaMA）包装成带 LoRA 插件的模型，用于只微调指定模块。 动态替换 nn.Linear 为 LoRA‑Linear
from model.moe_layer_llama import MoeBlock_RS#MoeBlock_RS：门控混合专家（MoE）结构的实现，用于根据聚类结果选择解释模型的不同“专家”。
from model.vamoe import Vmoe_llama3#Vmoe_llama3：整个“解释生成模块”的模型类，封装了 LLaMA 主干 + 门控 MoE + 用户/物品嵌入 + Prompt 头部等。

# vMF 聚类库（作者自定义实现）
from vmfmix.vmf import VMFMixture
# 数据集与模型组件
from dsvae.model import Decoder, VMFMM, Encoder
# 在模型加载之后、DeepSpeed 初始化之前，设置 MoE 模块为叶节点

try:
    from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss, get_load_balancing_loss
    from megablocks.layers.arguments import Arguments as MoEArgs
except ImportError:
    log.warning(f"Megablocks not installed. If you want to train MoEs, install with `pip install megablocks`.")
# 1. LoRA‑LLaMA 解释模型训练函数
def train(model, train_dataset, eval_dataset, tokenizer, epoch, checkpoint_dir, args):
    # 使用自定义 RecTrainer（继承 HF Trainer）可记录 gate 激活
    trainer = RecTrainer(
        model             = model,
        train_dataset     = train_dataset,  
        eval_dataset      = eval_dataset,
        tokenizer         = tokenizer,
        data_collator     = DataCollatorForSeq2Seq(#这是 Hugging Face 提供的一个函数，用于处理不定长序列的 padding 和 batch 对齐，常用于生成式任务
            tokenizer     = tokenizer,
            padding       = True,# 生成模型需要 pad，并保持 labels 对齐
        ),
        save_lora         = True,# 控制是否保存 LoRA 微调结果
        args = TrainingArguments( #是 Hugging Face 官方定义的一个类，用来指定训练超参数。

            output_dir                     = checkpoint_dir,#模型和检查点保存位置
            save_strategy                  = 'steps',#每隔 save_steps 步保存一次模型
            save_steps                     = 1000,
            per_device_train_batch_size    = 1,# 真正 batch=1 → 配合累积,用小 batch(=1) 走前向，把梯度攒够再更新，可“模拟”更大批次。
            learning_rate                  = 3e-5,
            num_train_epochs               = 1,
            gradient_accumulation_steps    = 16,#16 步累积一次梯度，相当于扩大 batch size,把 16 个小批次的梯度相加，再做一次优化器 step() 更新参数。
            # --------- logging arguments --------- #
            logging_strategy               = 'steps',
            logging_steps                  = 10,#	每 10 步记录一次训练日志
            report_to                      = 'tensorboard',#输出日志到 TensorBoard
            save_safetensors               = True,# 安全且可线上加载

            max_grad_norm                  = 0.3,# 避免梯度爆炸
            gradient_checkpointing         = True,	#启用梯度检查点，节省显存
            deepspeed                      = args.deepspeed,  
            bf16                           = True  # 新增：强制使用 bfloat16 精度训练
        )
    )

    print(len(trainer.train_dataset['input_ids'][0]),len(trainer.train_dataset['labels'][0]))#打印第一个样本的输入和标签长度，帮助检查数据是否处理正确。
    print('start {} training!'.format(args.dataset))
    trainer.train()

    print('{} training done!'.format(args.dataset))

    # ====================== save model ===================== #
    # trainer.save_model(checkpoint_dir)
    print('{} model saved!'.format(args.dataset))

# 2. 主入口 —— 全流程    
console = Console()
if __name__ == '__main__':
     # ---------------- CLI 参数 ----------------
    parser = argparse.ArgumentParser(description='VMoe_Rs')
     # 数据路径与名称
    parser.add_argument('--dataset', type=str, default='Amazon',
                        help='dataset name, ex: Amazon, Yelp, TripAdvisor')  #"数据集名称"   
    parser.add_argument('--data_path', type=str, default='/home/mail/2023t3/u430201701/hxproject/GavaMOE-vmf/datasets/Amazon/reviews.pickle',
                        help='data path') #"评分数据根目录"           
    parser.add_argument('--index_dir', type=str, default='/home/mail/2023t3/u430201701/hxproject/GavaMOE-vmf/datasets/Amazon/1',
                        help='dataset index file')  #"用户/物品索引文件夹"   
     # VAE 预训练 / 聚类 参数
    parser.add_argument('--pretrain_epochs', type=int, default= 300,#150
                        help='epoch of pretrain GMM')  #"VAE 预训练 epoch"   
    parser.add_argument('--latent_dim', type=int, default = 16,#128改成16
                        help='latent dim')    #     "隐空间维度"
    parser.add_argument('--embedding_size', type=int, default = 768,
                        help='user-item embedding size')    #"用户/物品嵌入维度"  
    parser.add_argument('--num_cluster', type=int, default = 2,
                        help='number of cluster')     
    parser.add_argument('--pretrain_model_path', type=str, default='/home/mail/2023t3/u430201701/hxproject/GavaMOE-vmf/meta-llama/Llama-3-8B-Instruct',
                        help='local path of llm')   #"LLaMA‑3 本地权重"
    parser.add_argument('--batch_size', type=int, default = 1024,#4096
                        help='batch size') 
    parser.add_argument('--cuda', action='store_true',default=True,
                        help='use CUDA')#启用 CUDA
    parser.add_argument('--pretrain_weight_save', type = str, default='/home/mail/2023t3/u430201701/hxproject/Copy_MegablocksGavaMOE-vmf/output/Amazon',
                        help='path to save the pretraining model')#保存 VAE 权重位置
    parser.add_argument('--cluster_epoch', type=int, default = 50,#30
                        help='epoch of cluster')#聚类训练 epoch
    parser.add_argument('--lr', type=int, default =  0.00001,#0.00001
                        help='Learning rate for training vae & gmm')#聚类阶段 Adam 学习率
    parser.add_argument('--output_dir', type = str, default = '/home/mail/2023t3/u430201701/hxproject/Copy_MegablocksGavaMOE-vmf/output/Amazon',
                        help='Explainable Model Training Results Storage Path')#解释模型输出目录
    parser.add_argument('--llm_epoch', type = int, default = 3, help='epoch of llm')#LoRA 训练 epoch
    parser.add_argument('--local_rank', type=int, default=-1, help='Deepspeed will pass this automatically')
    parser.add_argument('--deepspeed', type=str, default=None, help='deepspeed config path')

    args = parser.parse_args()

    # ========================================================  Config Setting  ======================================================== 
    seed = 105
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        device = 'cuda'
    else:
        device = 'cpu'

     # ---------------- 创建输出目录 ----------------
    if not os.path.exists(os.path.join(args.pretrain_weight_save, args.dataset)):
        os.makedirs(os.path.join(args.pretrain_weight_save,args.dataset), exist_ok=True)
        console.print(f'{args.dataset} Will be Save {os.path.join(args.pretrain_weight_save, args.dataset)}')
    
    # ---------------- 加载分词器 ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    tokenizer.pad_token = tokenizer.eos_token# # 因为LM 遇到 pad 会当作 eos

   # ---------------- 加载数据 ----------------
    console.print('Loading data...',style = 'bold green')
    max_text_length = 30 # prompt 截断防止超长
    
    corpus = DataLoader_Rs(args.data_path, args.index_dir, tokenizer, max_text_length)#self.train, self.valid, self.test, self.user2feature, self.item2feature 
    #self.user2feature记录了每个用户在训练集中提到过的特征（feature）{user_index: [feature1, feature2, ...]}
    #同样是一个字典，记录了每个物品被用户提及的特征列表，也是用于冷启动或解释生成
    n_user = len(corpus.user_dict)#表示用户或物品的种类数
    n_item = len(corpus.item_dict)
    
    # 打印参数
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)#----------------------------------------ARGUMENTS----------------------------------------
    for arg in vars(args):#遍历每一个参数名称（key）
        console.print('{:40} {}'.format(arg, getattr(args, arg)))#getattr(args, arg)获取该参数的值,'{:40} {}'.format(...)把参数名称左对齐，占 40 字符宽，然后打印对应值
    console.print(f"user num: {n_user} item num: {n_item}")#打印用户和物品数量
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    
    # ========================================================  Pretraining       ======================================================== 
    # ---------------- Phase 1：VAE+GMM 预训练 ----------------
    #以VAE 会把 ID 映射成一个稠密向量（比如 768 维），这些向量就包含了用户/物品的协同过滤语义（行为、偏好等）。
    vae_clu = Vae_Cluster_Es(n_user = n_user,n_item = n_item,args = args)
    # 保存模型结构，方便调参对比
    with open(os.path.join(args.pretrain_weight_save, args.dataset, args.dataset + '_output.txt'), 'w') as f:#把模型结构（vae_clu）写入文本文件，保存下来
        f.write(str(vae_clu))#str(vae_clu) 会自动调用 __str__() 方法，把模型结构（包括参数、层结构等）转换成字符串格式。
    vae_clu = vae_clu.to(device)
    vae_clu.pretrain(corpus = corpus, pretrain_epoch = args.pretrain_epochs)
    
    console.print(f'Pretraining finished....')#VAE 预训练完成
    # ========================================================  Cluster Training  ======================================================== 
    console.print(f'Cluster Training...')
    # vae_clu.cluster_training(corpus = corpus, cluster_epoch = 100)

     # ---------------- Phase 2：聚类精调 --------------------
    console.print(f'Start Cluster Training......', style='bold red')#聚类微调中…
    cluster_epoch = args.cluster_epoch  # 读取 CLI 超参；可在命令行 --cluster_epoch 调大/缩短，30
    epoch_bar = tqdm(range(cluster_epoch)) # tqdm 进度条，实时显示 epoch 进度
    data_loader = DataLoader(Dataset_Rs_Pytorch(corpus.train),batch_size = args.batch_size, shuffle = True)# DataLoader：包一层 Dataset_Rs_Pytorch，把原始列表转为张量；按 args.batch_size 随机打乱
    losses = [] 
    accuracies = []                                                                                           # user, item, rating , text, feature                                                      
    # lr=0.001 better   lr is important,2e-3 lead to posterior collapse 🤡
    optimizer = torch.optim.Adam(vae_clu.parameters(),lr = args.lr)#0.00001
    
    lr_s = StepLR(optimizer, step_size = 10, gamma = 0.5)# StepLR：每 10 epoch 把 lr 乘 0.5 ——> “阶梯衰减”
    print(f'len dataloader: {len(data_loader)}')# 查看一共多少 batch，便于估算显存/时间

    scale_factor_kl = 0.01 #β‑VAE：先小后大，防止 posterior collapse; # β-VAE 思想：先让 KL 系数小，聚焦重构；再逐步↑ 避免 collapse
    kl_increase = True# 是否启用 KL 退火
    for epoch in epoch_bar:
        # lr_s.step()
        epoch = epoch + 1
        loss_all = 0
        acccuracy_all=0
        losses_epoch = 0.
        best_val_loss = float('inf')
        print(f'scale_factor_kl is {scale_factor_kl}')
        for batch_index,(user, item, rating, _, _) in enumerate(data_loader):
            user = user.to(device)
            item = item.to(device)
            rating = rating - 1
            rating = rating.to(device)
            # compute elbo loss -> batch loss
            loss = vae_clu.vmfmm_Loss(user, item, rating, scale_factor_kl)# L′ = 0.1 × L，梯度变成 0.1 × ∂L/∂θ。批大小 4 K + 梯度累积 16	有效 batch ≈ 64 K，梯度天然更大；再配大 loss 系数，爆显存或数值溢出
            acccuracy=vae_clu.vmfmm_accuracy(user, item, rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss
            acccuracy_all += acccuracy
        # 每 5 epoch 增大 KL 系数 + 可视化 latent
        if epoch % 5 == 0: # scale_factor_kl 0.2 is better than 0.3
            # print('scale up scale_factor_kl')
            if kl_increase:#kl_increase = True# 是否启用 KL 退火
                scale_factor_kl += 0.005
                if scale_factor_kl >= 0.1:
                    scale_factor_kl = 0.1  
            #plot_latent(vae_clu, data_loader, args, epoch)
            #torch.save(vae_clu.state_dict(), os.path.join(args.pretrain_weight_save, args.dataset, args.dataset + '_' +f'cluster_{args.num_cluster}_epoch_{epoch}.pth'))
            #if vae_clu.encoder.r < 10.0:
                #vae_clu.encoder.r += 0.5
                #print(f'⏫ Increasing r: now {vae_clu.encoder.r:.2f}')
        plot_latent(vae_clu, data_loader, args, epoch)
        # 计算当前 epoch 的平均 loss（虽然这里没用到，用于 early-stop 或打印）
        losses_epoch = loss_all.item() / len(data_loader)  #losses_epoch
        # 保存最佳权重（这里用 train loss 当 proxy）
        if losses_epoch < best_val_loss:      
            best_val_loss = losses_epoch
            torch.save(vae_clu.state_dict(), os.path.join(args.pretrain_weight_save, args.dataset, args.dataset + '_' +f'cluster_{args.num_cluster}_best_weight.pth'))
            print(f'Saving Best Pretraining Model for loss {best_val_loss}')

        lr_s.step()
        print(f'Epoch {epoch} Loss: {loss_all.item() / len(data_loader)}')
        print(f'Epoch {epoch} Accuracy: {acccuracy_all/ len(data_loader)}')
        losses.append(loss_all.item() / len(data_loader))
        accuracies.append(acccuracy_all.item() / len(data_loader))

        vae_clu.plot_loss_curve(losses, title=f'Cluster Training Loss Curve for {args.dataset}',save_path= os.path.join(args.pretrain_weight_save,args.dataset, args.dataset +'_'+ f'loss_cluster_{args.num_cluster}.png'))
        vae_clu.plot_accuracy_curve(accuracies, title=f'Cluster Training Accuracy Curve for {args.dataset}',save_path= os.path.join(args.pretrain_weight_save,args.dataset, args.dataset +'_'+ f'accuracy_cluster_{args.num_cluster}.png'))

        # 每个 epoch 也保存一次（冗余但稳妥）
        torch.save(vae_clu.state_dict(), os.path.join(args.pretrain_weight_save, args.dataset, args.dataset + '_' +f'_cluster_{args.num_cluster}_epoch_{epoch}.pth'))
    console.print(f'Explaination Generate Training Start......',style = 'bold green')
    # WHY: 聚类微调完成 → 进入 “解释生成” 阶段（LoRA-LLaMA 训练）
    # ========================================================  Explaination Generate Training  ======================================================== 
    # construct Huggingface Dataset(把你之前的数据（如 corpus.train）从 Python 列表（List[Dict]）格式 → 转为 HuggingFace 支持的 Dataset 对象格式。)
    # ================= Phase 3‑A：构建 HF Dataset =======================
     # 设置缓存路径，避免每次都重新 Tokenize → 提高调试 & 训练效率
    cache_dir = os.path.join("/home/mail/2023t3/u430201701/hxproject/GavaMOE-vmf/datasets", "cached_datasets",args.dataset)  # 目录：output_dir/cached_datasets/
    os.makedirs(cache_dir, exist_ok=True)                         # 若目录不存在则创建
    train_cache = os.path.join(cache_dir, "train")                # 训练集缓存目录
    eval_cache = os.path.join(cache_dir, "eval")                  # 验证集缓存目录
    test_cache = os.path.join(cache_dir, "test")                  # 测试集缓存目录

    # 是否使用缓存机制，默认开启（由 argparse 参数控制）
    use_cache = True if hasattr(args, "use_cache") and args.use_cache else False

    # ---------------------- 若存在缓存，直接加载 ----------------------
    if use_cache and os.path.exists(train_cache):
        console.print("[green]Loading tokenized datasets from cache...[/green]")  # 提示信息
        train_dataset = load_from_disk(train_cache)  # 从磁盘恢复训练集
        eval_dataset = load_from_disk(eval_cache)    # 验证集
        test_dataset = load_from_disk(test_cache)    # 测试集

    # ---------------------- 否则重新处理 & 保存 ----------------------
    else:
        console.print("[yellow]Processing raw data and saving to disk...[/yellow]")
    # Step 1：将 PyTorch List[Dict] 数据结构转为 HuggingFace Dataset 对象
    train_dataset = TorchDataset2HuggingfaceDataset(corpus.train, cache_dir='')
    eval_dataset  = TorchDataset2HuggingfaceDataset(corpus.valid, cache_dir='')
    test_dataset  = TorchDataset2HuggingfaceDataset(corpus.test,  cache_dir='')
    # WHY：HuggingFace Trainer / map / filter / Dataloader 都依赖 Dataset 类。
    # 先统一格式，后面的 map、shuffle、batch 全免费获得

    # Step 2：对每条样本进行 Prompt 构造 + Tokenize（注意 batched=False）
    # Mapping the dataset 
    # bound to set batched to False, data process is not batched ref: prompt_precess.py examples['rating'] >=3 positive
    # -----------------------------------------------------------------------------
    # 对每条记录做 Prompt 重写 + Tokenize
    # -----------------------------------------------------------------------------
    print('Load the hf dataset...')#这个 map 操作会 添加新的字段（input_ids, attention_mask, labels），但它不会删除原始字段user, item, rating , text, feature
    train_dataset = train_dataset.map( # 把函数应用到每条样本
        Prompt_Process(tokenizer, 180),# ➜ 构造自然语言 prompt 并编码成 ID
        batched = False,               # ⚠️ 逐条处理，因函数里有 if/else
    )
    eval_dataset  = eval_dataset.map(
        Prompt_Process(tokenizer, 180),
        batched = False
    )
    test_dataset  = test_dataset.map(
        Prompt_Process(tokenizer, 180),
        batched = False
    )
    
    # WHY：
    # 1. Prompt_Process 把 (user,item,rating,text) 变成
    #    "User <u> likes Item <i> with 4 stars because …"
    #    同时生成 input_ids / labels（右移一格）
    # 2. max_len=180：截断过长文本，防止 GPU OOM
    # 3. 不做 batched=True：函数内部根据 rating 正负写不同模板

    # Step 3：保存至本地磁盘，下次可直接 load_from_disk 加快流程
    train_dataset.save_to_disk(train_cache)
    eval_dataset.save_to_disk(eval_cache)
    test_dataset.save_to_disk(test_cache)
    # 解码首条样本，肉眼检查 Prompt 是否正确
    console.print(tokenizer.decode(train_dataset['input_ids'][0]),style='bold green')#一条训练样本的 input_ids,decode() 会把 input_ids 转换为 人类可读的文本，
    # 由 VAE 预测 (user,item) 的聚类 → 保存为 gate 索引，供 MoE 路由
    #train_dataset = train_dataset.select(range(32))   # 只保留前 32 条样本
    

    # 由 VAE 预测 (user,item) 的聚类 → 保存为 gate 索引，供 MoE 路由
    train_cluster_index = save_gate_index(train_dataset, vae_clu)# 门控索引,专家路由：推荐解释模型是一个门控 MoE（Mixture of Experts）结构，gate index 指示每条样本应该交给哪个专家模型处理。
    # WHY：MoE 每个“专家”对应一个聚类；提前计算好索引，训练时 O(1) 查询。
    print(len(train_dataset['input_ids'][0]),len(train_dataset['input_ids'][1]))# double-check 长度一致
    # =============================================================================
    # Phase 3-B：构建 LoRA-增强的 LLaMA-3 MoE 模型
    # =============================================================================
    # ---------- 1) 配置 LoRA ----------
    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM, ## 因果语言模型
        target_modules = ['q_proj','v_proj','k_proj','o_proj','user_embed','item_embed'],#它告诉 LoRA 只对这几个模块加低秩可训练参数（LoRA 权重），而不是对整个模型都微调，减少训练资源消耗。
        modules_to_save = ['f3','f1','f2',               #MoE 专家网络	不同聚类的推荐解释
        'gate0','gate1','gate2','gate3','gate4',         #路由器（门控）	根据输入决定走哪个专家
        'user_proj','item_proj'],                        # 投影模块	将用户/物品嵌入映射到模型内部空间
        inference_mode = False,                          # 训练阶段
        #在 LoRA 中，我们冻结原始大模型的参数（比如 q_proj、v_proj 等矩阵），只训练低秩矩阵 A 和 B：W ′=W+α⋅A⋅B,W：原始预训练的权重（不动）,A⋅B：低秩的可学习参数（LoRA),α：一个缩放因子
        r = 8,                                           # LoRA rank=8
        lora_alpha = 16,                                 #LoRA 实际作用是：权重变成 W + αAB，其中 α = 16
        lora_dropout = 0.1                               #用来增加训练时的随机性，防止过拟合
)
    # ---------- 2) 载入 LLaMA-3 基础配置 ----------
    from model.config_llama3 import llama_config
    from model.config_moe import Moe_config
    config = llama_config
    config.torch_dtype = torch.bfloat16
    moe_config = Moe_config()
    print(config)# 打印确认 vocab_size / n_layer / n_head 等
    # ---------- 3) 取 VAE 训练好的 user/item 嵌入 ----------
    user_embeds = vae_clu.encoder.user_embeddings
    item_embeds = vae_clu.encoder.item_embeddings

    # WHY：把“评分重构”阶段学到的 ID 表示直接搬进 LLM，
    #      让文本解释模型天然带有协同过滤信息。
    # 转 bfloat16：节省 50% 显存，推理/训练更快；A100/H100 原生支持
    user_embeds = user_embeds.to(torch.bfloat16)
    item_embeds = item_embeds.to(torch.bfloat16)
    # ---------- 4) 构造 Vmoe_llama3 主干 ----------
    vmoe_llama3 = Vmoe_llama3(config = config,                       # 基础 GPT 框架
                              tokenizer = tokenizer,                 # 词表 → 方便自动 resize
                              gate_index_list = train_cluster_index, # 每条样本路由到哪个专家
                              user_embed = user_embeds,              # 冻结后的嵌入权重
                              item_embed = item_embeds, 
                              use_lora = False,
                              moe_config=moe_config)                      # 冻结 base，LoRA 覆盖, # 先不给 base 注入 LoRA，交给 PEFT 处理
    
    # ---------- 5) 用 PEFT 包装，加上 LoRA Adapter ----------
    model_llama3 = get_peft_model(vmoe_llama3,lora_config)

    # 释放 VAE 占用的显存
    vae_clu = vae_clu.to('cpu') #① 把 VAE 模型整体搬到 CPU；vae_clu 被移到 CPU 并删除 → 它的参数不再参与反向传播，L_ELBO 即使算出来也对梯度无贡献。
    del vae_clu# ② 删除 VAE 对象，彻底断开计算图
    torch.cuda.empty_cache()# ③ 清理 GPU 显存缓存
    
    print('Already Freeze the user item embedding...')
    # 打印可训练参数比例，确认 LoRA 生效
    print(model_llama3.print_trainable_parameters())
    
    # ================= Phase 3‑C：LoRA 训练 =============================
    #RecTrainer 继承了 HuggingFace Trainer，默认把 outputs.loss 当作总损失做 backward()，于是 Stage-2 训练 只对 LM Loss 更新 LoRA 和 MoE 参数。
    explain_checkpoint_dir = args.output_dir + '/explain'
    deepspeed.utils.set_z3_leaf_modules(model_llama3, [MoeBlock_RS])
    train(  
            epoch                   = args.llm_epoch, 
            model                   = model_llama3, 
            tokenizer               = tokenizer,
            train_dataset           = train_dataset,
            eval_dataset            = None,
            checkpoint_dir          = explain_checkpoint_dir,
            args                    = args
    )
    model_llama3.save_pretrained(explain_checkpoint_dir)
    print('Saved Model... && Training Done...')
