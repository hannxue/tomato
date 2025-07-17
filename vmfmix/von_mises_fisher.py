import math
import torch
from torch.distributions.kl import register_kl

from vmfmix.ive import ive
from vmfmix.hyperspherical_uniform import HypersphericalUniform


class VonMisesFisher(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,#loc 是方向中心，要求是实数张量
                       'scale': torch.distributions.constraints.positive}#scale 是集中度（κ），必须是正数；
    support = torch.distributions.constraints.real
    has_rsample = True#表示支持可导采样（重参数化）；
    _mean_carrier_measure = 0#是一些分布所需的标准属性，可暂不理会。

    #E[x]=A(κ)⋅μ,从这个分布中采样很多次，把所有样本加起来平均，得到的方向向量,μ：单位向量方向（均值方向）,𝐴(𝜅)一个小于等于1的缩放系数，决定“样本平均值”距离单位球面中心的远近
    #vMF 分布的均值是一个不再位于单位球面上的向量，而是从原点指向 μ 方向，但长度是 A(κ)，越集中越接近单位长度。
    @property#mean 近似为：均值方向 loc × 某个比例（由贝塞尔函数计算，决定集中程度）
    def mean(self):
        return self.loc * (ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale))

    @property#stddev 是 κ，在 vMF 分布中没有传统意义上的标准差，用集中度代替。
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None):
        self.dtype = loc.dtype
        self.loc = loc#单位向量，表示分布中心；
        self.scale = scale#浓度参数 κ，越大表示越集中；
        self.device = loc.device
        self.__m = loc.shape[-1]#维度；
        #构造一个形如 (1,0,0,…,0) 的单位向量，维度和 loc 一样。默认的采样中心方向。
        #z 的方向集中在 loc 周围	越靠近 loc 越可能被采样到（受 κ 控制），重参数采样公式并不容易直接实现，先在一个固定方向（比如 (1,0,...,0)，我们叫它“北极”）采样，把这个方向上的样本通过 旋转 变换对齐到用户指定的 loc 方向
        #Householder 变换是一种将向量 a 映射到向量 b 的正交线性变换，H(x)=x−2⋅⟨x,u⟩⋅u其中u= ∥a−b∥/a−b，这个变换可以把所有落在以 e1 为中心的球面分布的点旋转到 loc 上，不改变它们的模长或分布结构。
        self.__e1 = (torch.Tensor([1.] + [0] * (loc.shape[-1] - 1))).to(self.device)#用来做 Householder 变换的“北极向量”（默认朝向）；

        super(VonMisesFisher, self).__init__(self.loc.size(), validate_args=validate_args)#.loc.size() 是 (B, D)，表示这是一个**“批量分布”**（即你构造了 B 个 vMF 分布，每个有 D 维）。
    def sample(self, shape=torch.Size()):                                                #如果你没有传 shape，它就是 torch.Size()，即 []这表示对 每个分布采 1 个样本
        with torch.no_grad():                                                            #每个 vMF(𝜇𝑖,𝜅𝑖)分布是在单位球面 𝑆𝐷−1⊂𝑅𝐷上的概率分布。
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        #输入 shape 用于指定采样的样本数量。如果传入的是整数，会转换成 torch.Size 类型。
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        #采样主方向维度的 w，对应的是你在 loc 方向上的投影分量，值在 [-1, 1] 之间，控制采样点距离 loc 的“靠近程度”，w ≈ 1：采样点靠近方向中心（μ），w ≈ 0：采样点靠近赤道，w ≈ -1：远离方向中心
        # 根据当前球面的维度 m 决定采用哪种方式采样径向变量 w：
        #如果是在 3 维空间（S²），使用优化的 __sample_w3 方法；
        #否则使用更通用但效率较低的 __sample_w_rej 拒绝采样（rejection sampling）方法。
        w = self.__sample_w3(shape=shape) if self.__m == 3 else self.__sample_w_rej(shape=shape)
        #采样正交方向的向量 v，这是采样“球面上除了主方向以外的分量”，也就是正交于 (1, 0, ..., 0) 的单位向量。
        #先生成 shape 为 [B, D] 的高斯向量（零均值正态），.transpose(0, -1)[1:] 去掉第一维（也就是 (1, 0, ..., 0) 的方向），再转回来，得到 v.shape = [B, D-1]
        v = (torch.distributions.Normal(0, 1).sample(
            shape + torch.Size(self.loc.shape)).to(self.device).transpose(0, -1)[1:]).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)#归一化，确保单位模
        #拼成单位向量 x（方向在 (1,0,...,0) 附近）
        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))#w: 正向主轴分量（靠近中心
        x = torch.cat((w, w_ * v), -1)#w_ * v: 正交方向上分量（控制围绕中心的散布）；拼接之后的 x：是一个完整的单位向量，shape 是 [128, m]，但是方向是围绕 (1, 0, ..., 0) 的，不是围绕你真正的 loc
        # 第五步：用 Householder 旋转将 x 转到 loc
        z = self.__householder_rotation(x)#用 Householder 把 x 转到 loc，最终 z 就是你要的采样点！

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        #希望采样的“样本数量”与 κ（scale）参数的 shape 拼接，生成最终采样张量的 shape。
        shape = shape + torch.Size(self.scale.shape)#shape = torch.Size([32])          # 你希望采样 32 个样本
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = 1 + torch.stack([torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0).logsumexp(0) / self.scale
        return self.__w

    def __sample_w_rej(self, shape):
        #采样一个变量 w，它是：在单位球面上，沿着主方向 μ 的分量，即w=cos(θ)∈[−1,1]，用于构造球面采样点。
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)#计算常数 𝑐=根号下[4𝜅2+(𝑚−1)2],这是后续用于构造拒绝采样 proposal 的参数
        b_true = (-2 * self.scale + c) / (self.__m - 1)#真实计算的 b，用于构造 proposal 分布的偏移项

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)#b_app 是 Taylor 近似下的简化版（当 κ 非常大时，避免数值精度问题）
        #控制从 κ = 10 到 11 之间的“平滑切换”系数 s，s 介于 0~1，用于在 b_app 和 b_true 之间平滑插值
        #scale < 10 → s = 0 → 使用 b_true；scale > 11 → s = 1 → 使用 b_app
        s = torch.min(torch.max(torch.tensor([0.], device=self.device),
                                self.scale - 10), torch.tensor([1.], device=self.device))
        #得到最终使用的 b，在 b_app 和 b_true 之间平滑插值，避免 κ 较大时数值不稳定
        b = b_app * s + b_true * (1 - s)
        #这是 Wood 采样中用于 proposal 的中间变量 a
        a = (self.__m - 1 + 2 * self.scale + c) / 4
        #这是拒绝采样中的 normalization 常数 d，用于判断是否接受某个样本
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)
        #简单的 debug 检查，防止 a 为 NaN（κ 特别大时可能出错）
        if torch.isnan(a).any():
            print(1)
        #真正执行拒绝采样循环 __while_loop(...) 来采样 (e, w)，并将结果缓存
        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape)
        return self.__w#返回最终采样得到的 w 值，shape 一般是 [B, 1]，用于合成球面采样向量

    #拒绝采样（Rejection Sampling），从一个容易采样的“提议分布”中采样，然后根据概率决定是否接受该样本。
    def __while_loop(self, b, a, d, shape):

        b, a, d = [e.repeat(*shape, *([1] * len(self.scale.shape))) for e in (b, a, d)]
        w, e, bool_mask = torch.zeros_like(b).to(self.device), torch.zeros_like(
            b).to(self.device), (torch.ones_like(b) == 1).to(self.device)

        shape = shape + torch.Size(self.scale.shape)

        count = 0
        while bool_mask.sum() != 0:
            e_ = torch.distributions.Beta((self.__m - 1) / 2, (self.__m - 1) /
                                          2).sample(shape[:-1]).reshape(shape).to(self.device)
            u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1) * t.log() - t + d) > torch.log(u)
            reject = ~accept if torch.__version__ >= "1.2.0" else 1 - accept

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
            count += 1
            # if count > 10:
            #     print(1)
        # print(count)
        return e, w

    def __householder_rotation(self, x):
        u = (self.__e1 - self.loc)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        output = - self.scale * ive(self.__m / 2, self.scale) / ive((self.__m / 2) - 1, self.scale)

        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)

        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = - ((self.__m / 2 - 1) * torch.log(self.scale) - (self.__m / 2) * math.log(2 * math.pi) - (
            self.scale + torch.log(ive(self.__m / 2 - 1, self.scale))))

        return output.view(*(output.shape[:-1]))


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return - vmf.entropy() + hyu.entropy()
