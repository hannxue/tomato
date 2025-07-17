# -*- coding: utf-8 -*-
"""
解释性注释版：Encoder / Decoder / VMFMM
================================================
本文件对用户提供的 PyTorch 实现加入了**逐行中文注释**，并补充了
设计原理说明，便于后续阅读、修改和调试。重点说明如下：

1. **Von Mises–Fisher (vMF) 分布**
   - 用于建模单位超球面上的方向数据（embedding 单位化后位于 S^{D-1} 上）。
   - 参数 `mu` 表示均值方向；`kappa` (此处变量名为 *k*) 控制集中度，数值越大越集中。

2. **Encoder**
   - 输出 `(mu, kappa)` 参数，再从 vMF 分布中 *可微分* 地重参数化采样得到隐变量 *z*。
   - 采样的 *z* 与聚类中心同样位于单位球面，方便后续使用 vMF 混合模型进行聚类。

3. **VMFMM (vMF Mixture Model)**
   - 类似高斯混合模型 (GMM)，但组件分布更换为 vMF，以更契合方向数据。
   - 这里的 `vmfmm_Loss` 即对应 ELBO 中关于聚类先验部分的对数似然上界推导。

4. **初始化与软限制**
   - 使用 `softplus + r` 保证 kappa ≥ r > 0，避免过分尖锐的集中度导致数值不稳定。
   - 对中心 `mu` 做 *unit‑norm* 正则化，保证其依然落在单位球面上。

**注意：**所有注释以 `# 中文注释 …` 或多行 docstring 的形式直接嵌入代码。
如需进一步细化，请在 ChatGPT 对话中指出待补充的行号或关键词。
"""
try:
    import os
    import sys
    import math

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # ----------------------------- 项目内部依赖 -----------------------------
    from dsvae.utils import init_weights, d_besseli, besseli# 权重初始化工具，默认：正态分布 + Kaiming 变体  # 第一类修正贝塞尔函数 I_v(x) 的一阶导数，用于期望计算# 第一类修正贝塞尔函数 I_v(x)
    from dsvae.config import DEVICE# 读取全局设备配置 ("cuda" / "cpu")
    from vmfmix.von_mises_fisher import VonMisesFisher, HypersphericalUniform # 可重参数化 vMF 分布实现 # 单位球面上的均匀分布（此处未用到）

except ImportError as e:
    # 捕获导入错误，并抛出更友好的信息
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    在 `nn.Sequential` 中插入视图变换层的工具类
    """

    def __init__(self, shape=[]):
        """参数
        ----------
        shape : tuple 或 list
            目标形状 (不包含 batch 维)。
        """
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        # `*self.shape` 会解包为多维数，配合 batch size 重新调整 tensor 形状
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # 让 `print(model)` 时显示额外信息，便于调试
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.shape
        )


class Decoder(nn.Module):
    """反卷积解码器 (generator)。

    将球面隐变量 *z* 投影回图像空间 (1×28×28)。
    - 先全连接到一个高维向量，再 reshape 成 CNN 特征图，
      最后经 2 层 ConvTranspose2d 得到重构图像。
    """

    def __init__(self, latent_dim=50, x_shape=(1, 28, 28), cshape=(128, 7, 7), verbose=False):
        super(Decoder, self).__init__()
        # ------------------------- 参数缓存 -------------------------
        self.latent_dim = latent_dim         # 隐空间维度 (= Encoder 输出维度)
        self.ishape = cshape                  # reshape 后 CNN feature map 形状
        self.iels = int(np.prod(self.ishape))  # flatten 长度
        self.x_shape = x_shape                 # 最终输出图像形状
        self.output_channels = x_shape[0]
        self.verbose = verbose
        # ------------------------- 主干网络 -------------------------
        # 线性 -> ReLU -> 线性 -> ReLU -> Reshape -> 反卷积*2 -> Sigmoid
        # Sigmoid 让像素值限定在 [0, 1]，适合二值交叉熵 (BCE) 重构误差
        self.model = nn.Sequential(
            # ► 全连接阶段
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(True),

            nn.Linear(1024, self.iels),
            nn.ReLU(True),
            #
            Reshape(self.ishape),
        )
        self.model = nn.Sequential(
            # ► 卷积上采样阶段
            self.model,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(64, self.output_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
         # 初始化权重（Kaiming Normal 等），保持训练稳定
        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        """参数 x 为 shape=(B, latent_dim) 的隐变量。"""
        gen_img = self.model(x)
        # 返回形状统一的图像 (B, C, H, W)
        return gen_img.view(x.size(0), *self.x_shape)


class Encoder(nn.Module):
    """卷积编码器。

    结构设计遵循 *DCGAN* 风格：Conv → ReLU → Conv → ReLU → Flatten → FC → ReLU。
    额外输出：
        - `mu`      : 方向向量 (归一化到单位球面)
        - `k` (κ)   : 集中度，softplus 保证非负，并加常数 r (默认 80) 做下限
        - `z`       : 从 vMF(μ, κ) 重参数化采样的隐向量
    """

    def __init__(self, input_channels=1, output_channels=64, cshape=(128, 7, 7), r=80, verbose=False):
        super(Encoder, self).__init__()

        self.cshape = cshape
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.output_channels = output_channels# 隐空间维度 (= latent_dim64)
        self.input_channels = input_channels
        self.r = r               # κ 下限，防止 collapse
        self.verbose = verbose
        # ------------------------- 主干网络 -------------------------
        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.ReLU(True),
        )
        # ► 输出 μ, κ 分支
        self.mu = nn.Linear(1024, self.output_channels)#D 维单位向量，表示平均方向；
        self.k = nn.Linear(1024, 1)  # • κ (“kappa” / “k”) 是 标量，表示该方向上的集中度（κ 越大，样本越“挤”在 μ 附近）。


        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        """前向传播。
        返回 (z, mu, kappa)。
        """
        # ----- 1) 平均方向 μ -----
        x = self.model(x)
        mu = self.mu(x)
        # ----- 2) 集中度 κ -----
        # softplus 保证 κ > 0；加常数 r 做『地板』防止早期 κ 太大 or 为 0
        # We limit kappa to be greater than a certain threshold, because larger kappa will make the cluster more compact.
        k = F.softplus(self.k(x)) + self.r# shape = [B, 1]， 每个样本一个 κ，标量
        # ----- 3) 重参数化采样 z ~ vMF(μ, κ) -----
        # VonMisesFisher 类实现了 `rsample()`，因此可反向传播
        mu = mu / mu.norm(dim=1, keepdim=True) # 单位化，落在 S^{D-1}
        z = VonMisesFisher(mu, k).rsample()# (B, D)#如果你没有传 shape，它就是 torch.Size()，即 []这表示对 每个分布采 1 个样本

        return z, mu, k


class VMFMM(nn.Module):
    """vMF 混合模型 (可学习参数)。

    相当于在隐空间引入『聚类先验』：
        p(z, c) = π_c * vMF(z; μ_c, κ_c)
    其中 π_c, μ_c, κ_c 为可学习的组件参数。
    """

    def __init__(self, n_cluster=10, n_features=10):
        super(VMFMM, self).__init__()

        self.n_cluster = n_cluster # 聚类数 (= K)
        self.n_features = n_features# 隐空间维度 (= D)
        # ---- 参数初始化 ----
        # π_c：初始化为均匀分布 (1/K)
        # μ_c：随机向量 → 单位化，每簇均值 # κ_c：在区间 [1, 5] 均匀初始化（数值过大训练难，过小分布过于扁平）
        mu = torch.FloatTensor(self.n_cluster, self.n_features).normal_(0, 0.02)
        self.pi_ = nn.Parameter(torch.FloatTensor(self.n_cluster, ).fill_(1) / self.n_cluster, requires_grad=True)
        self.mu_c = nn.Parameter(mu / mu.norm(dim=-1, keepdim=True), requires_grad=True)
        self.k_c = nn.Parameter(torch.FloatTensor(self.n_cluster, ).uniform_(1, 5), requires_grad=True)
    # ------------------------------------------------------------------
    # 推断 (E‑step)：给定 z 计算责任概率 γ_ic，再取 argmax 得离散标签
    # ----------------------------------------------------------------
    def predict(self, z):
        """给定隐向量 z，输出硬聚类标签 (numpy array)。"""
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
        for c in range(self.n_cluster):
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
    def vmfmm_Loss(self, z, z_mu, z_k):
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

        det = 1e-10 # 防止 log(0)
        pi = self.pi_         # (K,)
        mu_c = self.mu_c      # (K, D)
        k_c = self.k_c        # (K,)

        D = self.n_features
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
        Loss = torch.mean((D * ((D / 2 - 1) * torch.log(z_k) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, z_k)) + e_k_mu_z_new)))

        # e_log_z_c
        Loss -= torch.mean(torch.sum(yita_c * (
                D * ((D / 2 - 1) * torch.log(k_c) - D / 2 * math.log(math.pi) - torch.log(besseli(D / 2 - 1, k_c)) + e_k_mu_z)), 1))

        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / yita_c), 1))
        return Loss
