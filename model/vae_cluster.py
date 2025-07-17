'''
    vae_cluster.py: 
        1、 Pretraining VAE to get prior#预训练 VAE（变分自编码器）以获得高质量潜在表示和 GMM 先验
        2、design Encoder and Decoder for rating construct# 定义 Encoder / Decoder 结构，用于重构离散评分 (1–5 stars)
        3、design elbo loss#定义基于 GMM 的 ELBO 损失 (重构项 + KL 项)
'''
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'#  # 限制 BLAS 线程，防止服务器过载
# os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
import sys
import math
import time
import torch
import torch.nn as nn       
import numpy as np
from rich.console import Console
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch.nn.functional as F
from utils.pepler_dataloader import DataLoader_Rs
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader,Dataset
import itertools
from utils.pepler_dataloader import Dataset_Rs_Pytorch
from utils.lr_utils import WarmUpLR
from rich.progress import BarColumn, Progress
from rich.live import Live
from rich.console import Console
from tqdm import tqdm
import datetime

# ----------------------------- 项目内部依赖 -----------------------------
from dsvae.utils import init_weights, d_besseli, besseli# 权重初始化工具，默认：正态分布 + Kaiming 变体  # 第一类修正贝塞尔函数 I_v(x) 的一阶导数，用于期望计算# 第一类修正贝塞尔函数 I_v(x)
from dsvae.config import DEVICE# 读取全局设备配置 ("cuda" / "cpu")
from dsvae.model import VMFMM
from vmfmix.von_mises_fisher import VonMisesFisher, HypersphericalUniform # 可重参数化 vMF 分布实现 # 单位球面上的均匀分布（此处未用到）
# vMF 聚类库（作者自定义实现）
from vmfmix.vmf import VMFMixture
console = Console()



'''
    Encoder: map user-item pair into latent space
    # --------------------------------------------------------------------------------------
    # Encoder: 把 [user, item] 嵌入拼接后送进两层 TransformerEncoder，
    #          输出潜在分布的均值 μ 和 log σ²。
    # WHY:
    #   * 用户/物品交互模式可能非线性，Transformer 的自注意力能学习高阶关系。
    #   * 将 μ, log σ² 直接回归，可在 KL 项里与 GMM 先验闭式计算。
    # --------------------------------------------------------------------------------------
'''
class Encoder(nn.Module):
    def __init__(self,n_user,n_item,embedding_size,latent_dim, r=10.0):
        super(Encoder,self).__init__()
        self.user_embeddings = nn.Embedding(n_user, embedding_size)#embedding_size768
        self.item_embeddings = nn.Embedding(n_item, embedding_size)

        self.user_embeddings.weight.data.uniform_(-0.2, 0.2)
        self.item_embeddings.weight.data.uniform_(-0.2, 0.2)
        # 两层 Transformer Encoder：d_model = user+item 拼接后的维度
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = embedding_size * 2, nhead = 4, dim_feedforward = 512)#每个位置的前馈神经网络中隐藏层的维度是 512（默认使用两个线性层）
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = 2)#定义了一个包含两层的 TransformerEncoder，每层都是注意力 + 前馈网络的结构。
        self.r = r  # <-- 变为动态可调的 r
        # mu and sigma
        ##self.mu_l = nn.Linear(embedding_size * 2, latent_dim)
        ##self.logvar_l = nn.Linear(embedding_size * 2, latent_dim)
        # ► 输出 μ, κ 分支
        self.mu = nn.Linear(embedding_size * 2, latent_dim)#D 维单位向量，表示平均方向；
        self.k = nn.Linear(embedding_size* 2, 1)  # • κ (“kappa” / “k”) 是 标量，表示该方向上的集中度（κ 越大，样本越“挤”在 μ 附近）。

    def forward(self, user, item):
        user_embedding = self.user_embeddings(user)
        item_embedding = self.item_embeddings(item)
        ui_embedding = torch.cat([user_embedding, item_embedding], dim=1)
        # ① 预归一化+dropout 可显著稳住数值
        ui_emb   = F.dropout(ui_embedding, p=0.1, training=self.training)
        ui_emb   = F.layer_norm(ui_emb, ui_emb.shape[-1:])
        ui_emb = ui_emb.unsqueeze(0)  #(1, B, 768*2)
        ui_out = self.transformer_encoder(ui_emb) #(1, B, 768*2)
        ui_out = ui_out.squeeze(0)  #  # (B, 768*2)
        #从这里改
        mu = self.mu(ui_out)
        # ----- 2) 集中度 κ -----
        # softplus 保证 κ > 0；加常数 r 做『地板』防止早期 κ 太大 or 为 0
        # We limit kappa to be greater than a certain threshold, because larger kappa will make the cluster more compact.
        raw_k_in = torch.clamp(self.k(ui_out), min=-15.0, max=15.0)   # 限幅 → 避免 exp(±∞)
        k = F.softplus(raw_k_in)+self.r  # <-- 变为动态可调的 r
        k = torch.clamp(k, max=50.0)
        #print(" k min/max:", k.min(), k.max())
        if torch.isnan(k).any(): raise ValueError("NaN in k")
        # ----- 3) 重参数化采样 z ~ vMF(μ, κ) -----
        # VonMisesFisher 类实现了 `rsample()`，因此可反向传播
        mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-8) # 单位化，落在 S^{D-1}
        if torch.isnan(mu).any(): raise ValueError("NaN in mu")
        #print("mu", mu)
        z = VonMisesFisher(mu, k).rsample()# (B, D)#如果你没有传 shape，它就是 torch.Size()，即 []这表示对 每个分布采 1 个样本
        return z, mu, k
'''
    Decoder: decoder the rating
    # --------------------------------------------------------------------------------------
    # Decoder: 把潜在向量 z 映射回 5 级评分 logits。
    # WHY: 评分是离散 1-5，需要 5-way softmax；直接用多层感知机足够。
    # --------------------------------------------------------------------------------------
'''
class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 5))# 5 类 logits
    def forward(self,z):
        z = self.fc1(z)# (B, 5)
        return z

# --------------------------------------------------------------------------------------
# Vae_Cluster_Es: 整体 VAE + GMM 模型
#     * 先用 VAE 预训练 ——> 得到潜在表示 Z
#     * 用高斯混合初始化 GMM 先验 (φ, μ_c, σ²_c)
#     * 训练时在 ELBO 中显式计算 KL(q(z|x) || p(z))，其中 p(z) 为 GMM
# WHY: “先自编码再聚类” 的变分深度聚类思想 (如 VaDE)。
# --------------------------------------------------------------------------------------
class Vae_Cluster_Es(nn.Module):
    '''
        args: 
            n_user: number of user
            n_item: number of item
            args  : parser parameter
    '''
    def __init__(self, n_user, n_item, args):
        super(Vae_Cluster_Es, self).__init__()

        self.encoder      = Encoder(n_user = n_user, n_item = n_item, embedding_size=args.embedding_size, latent_dim=args.latent_dim)
        self.decoder      = Decoder(latent_dim = args.latent_dim)
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ---- 参数初始化 ----
        # π_c：初始化为均匀分布 (1/K)
        # μ_c：随机向量 → 单位化，每簇均值 
        # κ_c：在区间 [1, 5] 均匀初始化（数值过大训练难，过小分布过于扁平）
        mu = torch.FloatTensor(args.num_cluster, args.latent_dim).normal_(0, 0.02)
        self.pi_ = nn.Parameter(torch.FloatTensor(args.num_cluster, ).fill_(1) / args.num_cluster, requires_grad=True)
        self.mu_c = nn.Parameter(mu / mu.norm(dim=-1, keepdim=True), requires_grad=True)
        self.k_c = nn.Parameter(torch.FloatTensor(args.num_cluster, ).uniform_(1, 5), requires_grad=True)
        # GMM 参数：先随机/零初始化，预训练后会被 GMM 拟合值覆盖
        ##self.phi          = nn.Parameter(torch.FloatTensor(args.num_cluster,).fill_(1) / args.num_cluster, requires_grad=True)# 混合权重
        ##self.mu_c         = nn.Parameter(torch.FloatTensor(args.num_cluster, args.latent_dim).fill_(0), requires_grad=True)# 每簇均值
        ##self.log_sigma2_c = nn.Parameter(torch.FloatTensor(args.num_cluster, args.latent_dim).fill_(0), requires_grad=True)# 每簇对角 log σ²
        self.args         = args


    def now_time(self):
        return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '
    def plot_loss_curve(self, losses, title='Pretraining Loss Curve', save_path='loss_curve.png', dpi=200):
        """保存训练曲线"""
        plt.figure(figsize=(10, 5))

        sns.lineplot(x = range(len(losses)), y = losses,marker='*',markerfacecolor='#F0988C', markersize=16, markevery=10)
        plt.gca().lines[0].set_color('#C76DA2')
        plt.gca().lines[0].set_linestyle('-')
        plt.gca().lines[0].set_linewidth(2.5)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    def plot_accuracy_curve(self, accuracies, title='Accuracy per Epoch', save_path='accuracy_curve.png', dpi=200):
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=range(len(accuracies)), y=accuracies, marker='o',markerfacecolor='#F0988C', markersize=16, markevery=10)
        plt.xlabel('Epoch')
        plt.ylabel('Clustering Accuracy (%)')
        plt.title(title)
        plt.grid(True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()


    def cluster_acc(self,Y_pred, Y):
        import scipy.io as scio
        from scipy.optimize import linear_sum_assignment  # 替代旧 linear_assignment
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i,j] for i,j in zip(*ind)])*1.0/Y_pred.size, w

    # 先用预训练把latent space训练好 得到先验分布
    # ==================================================================================
    # Phase-1: 仅优化 Encoder+Decoder —— 重构评分，目的是学出稳定潜在空间作为 GMM 先验
    # ==================================================================================
    def pretrain(self, corpus, pretrain_epoch):
        assert self.args.pretrain_weight_save is not None#"必须指定 --pretrain_weight_save"
        print(f'Start Pretraining !!!!!')
        # 如果没有预训练权重文件，才开始训练流程
        if not os.path.exists(os.path.join(os.path.join(self.args.pretrain_weight_save,self.args.dataset, self.args.dataset + f'_cluster_{self.args.num_cluster}_'  +'pretrain_weight.pth'))):
            warm = 1  # 预热 epoch 数
            Loss = nn.CrossEntropyLoss()        
            optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()),lr = 0.00015)# 0.00015
            train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [60, 120], gamma = 0.9) #learning rate decay# 学习率调度器：在第 60 和 120 epoch 时衰减,每次触发时，将学习率乘以 0.9（即下降 10%）
            # batch_size influences the Training Time in V100 32GB. ex: 81920 vs 1024, meanwhile, which influences the result of clustering
            # btw: using Adam optimizer, remember to set small batch size like 256,512
            # 训练数据加载器
            data_loader = DataLoader(Dataset_Rs_Pytorch(corpus.train),batch_size = 2048,shuffle = True)#，只使用了 corpus.train 来进行 VAE 的预训练。user, item, rating , text, feature
            iter_per_epoch = len(data_loader)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)# WarmUpLR：预热调度器，用于前 warm 个 epoch 内逐步提升学习率，防止训练初期不稳定
            print("================================================ Pretraining Start ================================================")
            epoch_bar = tqdm(range(pretrain_epoch))
            start = time.time()
            losses = []
            
            best_val_loss = float('inf') # 初始化最佳 loss
            endure_count = 0
            endure_count_break = 15#如果连续 15 次都没有提升，就停止训练
            for epoch in epoch_bar:
                total_sample = 0. 
                losses_epoch = 0.
                epoch = epoch + 1
                if epoch > warm:
                    train_scheduler.step(epoch)#在前 warm=1 个 epoch 中，我们使用的是 WarmUpLR，即学习率线性增长调度器,在第 warm+1 个 epoch 之后，才启用 MultiStepLR，根据训练轮数调整学习率；
                for batch_index,(user, item, rating, _, _) in enumerate(data_loader):
                    user = user.to(self.device)
                    item = item.to(self.device)
                    rating = rating - 1 #将原始评分转换成从 0 开始的类别索引,因为 nn.CrossEntropyLoss() 要求目标标签（target）是 从 0 开始的整数分类标签，所以需要对原始的评分 rating 执行 -1 操作。
                    rating = rating.to(self.device)
                    optimizer.zero_grad()
                    #这里改了
                    z, _, _  = self.encoder(user,item)
                    pre_rating = self.decoder(z)
                    loss = Loss(pre_rating, rating)
                    losses_epoch += loss
                    loss.backward()
                    optimizer.step()
                    if batch_index % 100 == 0:#表示每训练 100 个 batch 就记录一次当前的 loss 值，加入 loss 曲线中，方便后续可视化训练收敛过程。
                        losses.append(loss.item())
                    console.print(':thumbs_up: :pile_of_poo: Time:{time} Training Epoch: {epoch}/{all_epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                            loss.item(),
                            optimizer.param_groups[0]['lr'],
                            time = self.now_time(),
                            epoch=epoch,
                            all_epoch=pretrain_epoch,
                            trained_samples=batch_index * data_loader.batch_size + len(user),#训练进度统计：batch_index * batch_size 是之前处理过的样本数，len(user) 是当前 batch 的大小# 所以总的已处理样本数为 (前面 batch 的样本数 + 当前 batch 样本数)
                            total_samples=len(data_loader.dataset)
                    ), style="bold red")
                    total_sample += data_loader.batch_size
            
                if epoch <= warm:
                    warmup_scheduler.step()
                losses_epoch = losses_epoch / len(data_loader)
                if losses_epoch < best_val_loss:   #如果本轮训练（一个 epoch）的平均 loss（losses_epoch）小于当前记录的最佳 loss（best_val_loss），说明模型性能提升了。   
                    best_val_loss = losses_epoch
                    torch.save(self.state_dict(), os.path.join(self.args.pretrain_weight_save,self.args.dataset,self.args.dataset + f'_cluster_{self.args.num_cluster}_' + 'best_' +'pretrain_weight.pth'))
                    print(f'Saving Best Pretraining Model for loss {best_val_loss}')
                else:
                    endure_count += 1#如果当前 epoch 的 loss 没有变好，说明训练性能停滞，那就累加耐心计数器 endure_count。
                    console.print(self.now_time() + 'We are going to early stop..., Which is Harder...')
                    if endure_count == endure_count_break:
                        break
             
            finish = time.time() 
            print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))#打印整个训练所耗费的时间，帮助评估训练效率。
            #把 encoder 中 mu_l（均值分支）的参数 完整拷贝 到 logvar_l（对数方差分支）,模型初始化时令编码器的均值和方差相同（冷启动，避免差异太大导致训练不稳定）
            #通常用于预训练结束后，初始 GMM 聚类前，确保隐变量的均值和方差编码具有一致性，便于之后 GMM 的聚类稳定收敛。
            #self.encoder.logvar_l.load_state_dict(self.encoder.mu_l.state_dict())
            self.plot_loss_curve(losses, title='Pretrain Loss Curve',save_path=os.path.join(self.args.pretrain_weight_save,self.args.dataset,self.args.dataset + '_' + 'pretrain_loss.png'))
            
            Z = []
            with torch.no_grad():
                for user, item, rating, _, _ in data_loader:
                    user = user.to(self.device)
                    item = item.to(self.device)
                    #这里改了
                    z, mu, k   = self.encoder(user,item)
                    #assert F.mse_loss(z1, z2) == 0
                    Z.append(z)#例Z = [Tensor[2048, 10], Tensor[2048, 10], ..., Tensor[1024, 10]]
            # Z shape : batch,latent_dim
            Z = torch.cat(Z, 0).detach().cpu().numpy()#)#此时 encoder 输出的是确定性的 z（因为没有采样），torch.cat(Z, 0)	沿第 0 维（样本维）将多个 batch 拼接成一个大张量
          
            ##gmm = GaussianMixture(n_components = self.args.num_cluster, covariance_type='diag')#用 GMM 在隐空间 z 上聚类
            ##pre = gmm.fit_predict(Z)#返回的是一个整数向量，其中每个值是该样本所属的聚类标签（cluster ID），取值范围是 0 ~ num_clusters - 1。
            _vmfmm = VMFMixture(n_cluster=self.args.num_cluster, max_iter=100)#来源于 vmfmix.vmf.VMFMixture，是用来做初始聚类的工具。
            _vmfmm.fit(Z)
            pre = _vmfmm.predict(Z)
            

            #  可视化选取
            num_samples_per_cluster = 500
            indices = []
            for i in range(self.args.num_cluster):
                indices_in_cluster = np.where(pre == i)[0]#找出预测为第 i 类簇的样本索引。
                selected_indices = np.random.choice(indices_in_cluster, num_samples_per_cluster, replace=False)#从这些索引中**随机选出一定数量（如 500）**进行可视化，replace=False 表示不重复抽样。
                indices.extend(selected_indices)#indices.extend(...): 将每一簇中选中的样本索引添加进 indices 列表，最终用于拼接一组用于 t-SNE 可视化的子集。

            selected_Z = Z[indices]


            tsne = TSNE(n_components=3, init='pca', random_state=42)  
            Z_tsne = tsne.fit_transform(selected_Z)
            # —— 投射到单位球面，便于 3D 球面直观展示
            norms = np.linalg.norm(Z_tsne, axis=1, keepdims=True) + 1e-12
            Z_sph = Z_tsne / norms                   # (N, 3)
            
            selected_pre = pre[indices]#每个可视化点所属的聚类编号。
            
            fig = plt.figure(figsize=(14, 14))
            ax  = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(Z_sph[:, 0], Z_sph[:, 1], Z_sph[:, 2],c=selected_pre, cmap='viridis',  # 使用聚类标签进行着色
                     s=8, alpha=0.8, edgecolors='k', linewidths=0.2)

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
            #fig.colorbar(sc, ax=ax, shrink=0.6)  # 添加颜色条

             
            plt.title(f'Vis of Pretrain Latent Space for {self.args.dataset}') 
            plt.savefig(os.path.join(self.args.pretrain_weight_save,self.args.dataset, self.args.dataset + '_' + f'pretrain_latent_{self.args.num_cluster}.png'),dpi=300)

            print('VMFMixture Model Fit Done......')
            #这里改了
            ##self.phi.data = torch.from_numpy(gmm.weights_).cuda().float()
            ##self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            
            self.pi_.data = torch.from_numpy(_vmfmm.pi).cuda().float()
            self.mu_c.data = torch.from_numpy(_vmfmm.xi).cuda().float()
            self.k_c.data = torch.from_numpy(_vmfmm.k).cuda().float()
            # 注意这里取了log
            #self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())
            
            if not self.args.pretrain_weight_save:
                os.mkdirs(self.args.pretrain_weight_save)

            torch.save(self.state_dict(), os.path.join(self.args.pretrain_weight_save,self.args.dataset,self.args.dataset + f'_cluster_{self.args.num_cluster}_'  +'pretrain_weight.pth'))
            self.load_state_dict(torch.load(os.path.join(self.args.pretrain_weight_save,self.args.dataset,self.args.dataset + f'_cluster_{self.args.num_cluster}_'  +'pretrain_weight.pth')))
            print('loaded best pretrain weight')
        else:
            self.load_state_dict(torch.load(os.path.join(self.args.pretrain_weight_save,self.args.dataset, self.args.dataset + f'_cluster_{self.args.num_cluster}_'  +'pretrain_weight.pth')))
            print('already loaded')
    
    
    # ------------------------------------------------------------------
    # 推断 (E‑step)：给定 z 计算责任概率 γ_ic，再取 argmax 得离散标签
    # ----------------------------------------------------------------
    def predict(self,user, item):
        """给定隐向量 z，输出硬聚类标签 (numpy array)。"""
        z, _, _  = self.encoder(user, item)
        pi = self.pi_
        mu_c = self.mu_c
        k_c = self.k_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.vmfmm_pdfs_log(z, mu_c, k_c))

        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)
    # ------------------------------------------------------------------
    # 采样：指定某簇 (索引 k) 的条件下生成方向向量 
    # ------------------------------------------------------------------
    def sample_by_k(self, k, num=10):
        """从第 k 个组件 vMF(μ_k, κ_k) 采样 num 个 z。"""
        mu = self.mu_c[k:k+1]
        k = self.k_c[k].view((1, 1))
        z = None
        for i in range(num):
            _z = VonMisesFisher(mu, k).rsample()
            if z is None:
                z = _z
            else:
                z = torch.cat((z, _z))
        return z
    # ------------------------------------------------------------------
    # 辅助函数：批量计算 log p(z | c=k)对 z，计算它在每一个vmf分布 𝑐𝑘上的对数概率密度 log𝑝(𝑧𝑖∣𝑐𝑘)
    # ------------------------------------------------------------------
    def vmfmm_pdfs_log(self, x, mu_c, k_c):

        VMF = []
        for c in range(self.args.num_cluster):
            VMF.append(self.vmfmm_pdf_log(x, mu_c[c:c + 1, :], k_c[c]).view(-1, 1))
        return torch.cat(VMF, 1)
    
    """单组件 vMF 对数密度。
        公式：
            log f(z) = (D/2 - 1) log κ - D/2 log π - log I_{D/2-1}(κ) + κ μ^T z
    """
    @staticmethod
    def vmfmm_pdf_log(x, mu, k):
        D = x.size(1)
        log_pdf = (D / 2 - 1) * torch.log(k) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, k)) \
                  + x.mm(torch.transpose(mu, 1, 0) * k)#	torch.transpose(mu, 1, 0)(D, 1)	把均值方向 μ 从 (1, D) 转成列向量，便于后续矩阵乘，x.mm(...)：批量计算每个样本与 κ μ 的点积。
        return log_pdf
    # ------------------------------------------------------------------
    # 变分下界 (ELBO) 中关于聚类先验的期望项 (负号后做最小化)。
    # ------------------------------------------------------------------
    def vmfmm_Loss(self, user, item, rating, scale_factor_kl):
        """
        参数
        ----------
        z    : (B, D)   采样隐变量
        z_mu : (B, D)   Encoder 输出方向向量
        z_k  : (B, 1)   Encoder 输出集中度

        返回
        ------
        Loss : torch.Tensor 标量；值越小越好 (已取负期望)
        """
        L                  = 1                            # 采样次数；设 1 即可，省显存省时间
        z, z_mu, z_k = self.encoder(user, item)     # q(z|x) 均值 & log σ²
        loss_func          = nn.CrossEntropyLoss()        # 离散 5-级评分 → 交叉熵
        # start sampling   
        # ------------------------- (1) 重构项 ----------------------------------
        elbo_loss          = 0
        for l in range(L): 
            z = z.to(self.device)
            pre_rating = self.decoder(z)  # (B,5)
            elbo_loss += loss_func(pre_rating, rating)


        Loss           = elbo_loss /  L * self.args.embedding_size * 2#elbo_loss / L如果你每次采样多个 z（L>1），那要对它们的损失求平均。,* self.args.embedding_size * 2这是为了让 reconstruction loss 的量级和 KL 项的量级相对一致。
        det = 1e-10 # 防止 log(0)
        pi = self.pi_         # (K,)
        mu_c = self.mu_c      # (K, D)
        k_c = self.k_c        # (K,)

        D = self.args.latent_dim#self.n_features
        # ---------- 1) 责任概率 γ_ic = q(c|z) ----------
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.vmfmm_pdfs_log(z, mu_c, k_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))  # batch_size*Clusters # 归一化到概率

        # ---------- 2) E_q[κ_c μ_c^T z] ----------
        # dI_v/dκ / I_v(κ) ≈ E[μ^T z]，参见 vMF 的期望性质
        # batch * n_cluster
        e_k_mu_z = (d_besseli(D / 2 - 1, z_k) * z_mu).mm((k_c.unsqueeze(1) * mu_c).transpose(1, 0)) # (B, K)
        
        # ---------- 3) E_q[κ_z μ_z^T z] (self‑term) --------
        # batch * 1
        e_k_mu_z_new = torch.sum((d_besseli(D / 2 - 1, z_k) * z_mu) * (z_k * z_mu), 1, keepdim=True)

        # e_log_z_x
        kl = torch.mean((D * ((D / 2 - 1) * torch.log(z_k) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, z_k)) + e_k_mu_z_new)))

        # e_log_z_c
        kl -= torch.mean(torch.sum(yita_c * (
                D * ((D / 2 - 1) * torch.log(k_c) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, k_c)) + e_k_mu_z)), 1))

        kl -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / yita_c), 1))
        kl = kl * scale_factor_kl
        Loss += kl
        return Loss * 0.1
    def vmfmm_accuracy(self, user, item, rating):
       
        z, _, _ = self.encoder(user, item)
        tru=rating-1
        z = z.detach().cpu().numpy()
        tru = tru.detach().cpu().numpy()
        _vmfmm = VMFMixture(n_cluster=self.args.num_cluster, max_iter=100)#来源于 vmfmix.vmf.VMFMixture，是用来做初始聚类的工具。
        _vmfmm.fit(z)                         # 先训练聚类模型
        pre = _vmfmm.predict(z)              # 然后才能预测
        accuracy=self.cluster_acc(pre,tru)[0]*100
        return   accuracy

    # =============================================================================
    #  Vae_Cluster_Es.Elbo_loss
    #  ---------------------------------
    #  输入:
    #     user, item  : LongTensor，批量用户 / 物品索引
    #     rating      : LongTensor，真实评分 (0-4，对齐 CrossEntropyLoss)
    #     scale_factor_kl : float，β-VAE 的 KL 退火系数
    #
    #  输出:
    #     标量 Loss  —— 交叉熵重构项 + β·KL(GMM‖Posterior)
    #
    #  WHY 设计要点
    #  -------------
    #  1) 先采样 z 重构评分，得到   L_rec             （重构项）
    #  2) 再利用 GMM 先验闭式公式  L_KL              （正则项）
    #  3) β-退火 (scale_factor_kl) 逐渐↑，防止 posterior collapse
    #  4) 最后整体乘 0.1 与作者经验保持量级，利于 LR 调参
    # =============================================================================
    