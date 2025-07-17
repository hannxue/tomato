import os
import sys
import torch.nn as nn
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from model.vae_cluster import Vae_Cluster_Es
import warnings
warnings.filterwarnings('ignore')

from transformers import LlamaPreTrainedModel,AutoModelForCausalLM # 快速加载官方 Llama-3 权重
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions# 标准输出类型
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from model.moe_layer_llama import MoeBlock_RS # 自定义 MoE Block
from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss, get_load_balancing_loss
# -----------------------------------------------------------------------------
#  说明：MoeBlock_RS 内部实现了多专家 + 门控，
#       会在 `change_mlp4moe()` 中把 Llama 原生 MLP 替换掉
# -----------------------------------------------------------------------------

def z_loss_func(logits: torch.Tensor, z_loss_coeff: float = 1e-3):
    """Router logits 正则项，来自 ST-MoE 提升训练稳定性"""
    return torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff

class Vmoe_llama3(LlamaPreTrainedModel):
    # -------------------------------------------------------------------------
    #  ⬇️  构造函数：组装   ① 预训练 Llama-3 主干
    #                     ② 用 MoE 替换 feed-forward
    #                     ③ 注入用户/物品嵌入
    #                     ④ （可选）套 LoRA 低秩适配器
    # -------------------------------------------------------------------------
    def __init__(self, config, tokenizer, gate_index_list, user_embed, item_embed,moe_config,use_lora = False):
        super().__init__(config)
        # 1) 保存实参到对象里
        self.config = config
        self.gate_index_list = gate_index_list      #  每条训练样本对应的“聚类 → 专家索引”
        self.tokenizer = tokenizer
        self.use_lora = use_lora
        ### ← 新增：统一设备
        self.moe_config = moe_config
        # 2) 加载 Llama-3-8B-Instruct（bf16 节省显存）
        #    low_cpu_mem_usage=True 可边读边释放内存
        #你自己手动造了 config，但 又 让 from_pretrained() 去自动读取目录里的 config.json。两份 config 对冲
        self.model = AutoModelForCausalLM.from_pretrained("/home/mail/2023t3/u430201701/hxproject/GavaMoE/meta-llama/Llama-3-8B-Instruct",
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.bfloat16,
                                                    local_files_only=True,      # 防止再去 Hub 读另一份 config
                                                    config = self.config)#config = self.config
        ## 2️⃣ 读完以后，再手动把缺失字段补进去
        ##model.config.parallelization_style = ""        # 必须是可迭代对象
        ##model.config.num_cluster            = 4              # 你自定义的其它字段也可顺手补
        # 3) 用 MoEBlock_RS 替换每层的 MLP
        self.change_mlp4moe()
        print(self.model)   # 打印模型结构，确认替换成功
        # useless
        if self.use_lora:
            pass
        # ------- 4) 配置用户 / 物品嵌入 ----------    
        if user_embed is not None and item_embed is not None:
            # WHAT: 复用 VAE 里训练好的嵌入权重
            self.user_embed = user_embed
            self.item_embed = item_embed
            # WHY: 冻结复制，可在解释文本中携带协同过滤语义
            with torch.no_grad():  
                self.user_embed.weight.data.copy_(user_embed.weight.data)
                self.item_embed.weight.data.copy_(item_embed.weight.data)
        else:
            # replace different dataset # 若缺乏预训练嵌入，用随机 Embedding 兜底
            self.user_embed = nn.Embedding(9765, 768).to(torch.bfloat16)
            self.item_embed = nn.Embedding(6280, 768).to(torch.bfloat16)

        ## 把 768-d CF 向量 → 投射到 Llama hidden_size
        self.user_proj = nn.Linear(768, config.hidden_size).to(torch.bfloat16)#768->4096
        self.item_proj = nn.Linear(768, config.hidden_size).to(torch.bfloat16)
        # 用于统计 MoE 路由调用次数（debug）
        self.index_count = 0
        # transformers 父类收尾：初始化权重 & 注册 buffer
        self.post_init()
    # -------------------------------------------------------------------------
    #  把 Llama-3 每层的 `mlp` 替换为自定义 MoEBlock_RS
    # -------------------------------------------------------------------------
    def change_mlp4moe(self):
        # for the modulelist of llama
        count = 0 
        for block in self.model.model.layers:
            # WHAT: 原来 block.mlp 是 (Gate → SwiGLU → FC) 普通 FFN
            #       现在替换为 “门控 + K 个专家 + 聚合”
            block.mlp = MoeBlock_RS(config = self.config, 
                                    cluster_index_list = self.gate_index_list, 
                                    dataset_num = 100,
                                    moe_config=self.moe_config).to(torch.bfloat16)#  门控平滑超参
            print(f'already replace mlp to moe {count}')#print(f"[MoE 替换] layer-{idx} 已替换为 MoE MLP")
            count = count + 1
        return # 返回值只是占位#此后 self.model 的计算图已经变成“Attention + MoE-FFN”,forward() 里只管跑 已经替换后的模型；
        # model.base_model.model.model.model
    '''
        BATCH_SIZE must be 1, or forward func is wrong.
    '''

    # =========================================================================
    #  forward() —— 训练阶段前向
    #
    #  强假设：batch_size = 1。原因：
    #   1) 门控索引 gate_index_list[i] 对应样本 i；写批量逻辑更复杂
    #   2) 解释训练常用 batch=1 + 累积梯度，显存友好
    # =========================================================================
    def forward(
        self, 
        user, 
        item, 
        input_ids, 
        attention_mask,
        labels,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,                                # KV cache（推理时用）
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
        ):
        
        clear_load_balancing_loss()          # ←★ 只在这里清一次
        # ---------- 1) 统计 MoE 调用次数（debug） ----------
        # ⇩⇩⇩ ① 这里就把每一层 MoE 的游标都 +1 了
        for block in self.model.model.layers:
            block.mlp.cluster_index_count  = block.mlp.cluster_index_count + 1
        
        device = self.model.device
        batch_size = input_ids.size(0)
        # ---------- 2) 取用户/物品嵌入并投射 ----------
        user = torch.tensor(user).contiguous().to(device)
        item = torch.tensor(item).contiguous().to(device)

       
        user_embed = self.user_embed(user).to(device)
        item_embed = self.item_embed(item).to(device)
        
        self.user_proj = self.user_proj.to(device)
        self.item_proj = self.item_proj.to(device)
        #VAE 中提取了 user_embed 和 item_embed，但它们维度是 768（协同过滤学到的维度）而 LLaMA 需要的是 hidden_size = 4096 用 Linear(768 → hidden_size) 投影变换一下，才能喂给 LLaMA
        user_embeds = self.user_proj(user_embed)
        item_embeds = self.item_proj(item_embed)
        
        
        # # ITEM: 37032 USER: 14194
         # ---------- 3) 找到 prompt 中 <USER> / <ITEM> 的占位 token ----------
        #    - 14194 / 37032 是提前把特殊 token 加到 tokenizer 里的 id
        #    - 只能出现一次；否则报错#input_ids（batch,seq_len)
        user_index = torch.where(input_ids[0] == 14194)# input_ids[0]取出第一个样本的 token 序列；找出 <USER> 这个 token 出现在哪个位置，
        item_index = torch.where(input_ids[0] == 37032)# 找出 <ITEM> 的位置
        # 断言：必须找到了两个特殊 token
        assert user_index is not None and item_index is not None, "Indices must not be None"
        # 断言：每个 token 最多只能出现一次，否则不允许训练（避免重复替换出错）
        assert len(user_index[0]) < 2 and len(item_index[0]) < 2, "Indices must be less than 2"
        #将 tensor 里的位置取出并转为 Python int 类型：user_index[0] 是 tensor([1])，user_index[0][0] 是 tensor(1)，.item() 转成纯整数 → 1
        user_index = user_index[0][0].item()
        item_index = item_index[0][0].item()
        # ---------- 4) 获取 token 级嵌入 ----------
        # 把 prompt 中 <USER> 这个 token 对应的嵌入，替换成 user_embeds
        # 把 <ITEM> 对应的嵌入，替换成 item_embeds
        if self.use_lora:
            # 这是调用原始 LLaMA 的嵌入层，把 input_ids 转成 [batch_size, seq_len, hidden_size] 的向量序列。
            #结果就是：把一串 token id（整数序列）转成对应的向量（float tensor）
            #prompt_embeds.shape = [batch=1, seq_len, hidden_size]
            prompt_embeds = self.model.base_model.model.model.embed_tokens(input_ids)
        else:
            prompt_embeds = self.model.model.embed_tokens(input_ids)
        # ---------- 5) 把占位符替换成 CF 嵌入 ----------
        #原来的 <USER> 和 <ITEM> 是普通的 token embedding，没语义。
        #你现在用 VAE 训练好的用户/物品向量 来替换它们，嵌入里就有协同过滤的语义了。
        #这一步非常关键，等于让 LLaMA 模型在文本中“看到”了用户和物品的行为特征。
        prompt_embeds[0][item_index] = item_embeds#重要的是替换
        prompt_embeds[0][user_index] = user_embeds

        final_embed = prompt_embeds
        # ---------- 6) 喂给 Llama 做因果 LM  ----------
        #你不是给 input_ids，而是给 inputs_embeds，也就是嵌入向量。HuggingFace 会跳过 embed_tokens，直接用你构造的嵌入。此时模型就会根据这段 prompt + CF 语义，来生成目标文本。
        llm_output = self.model(inputs_embeds = final_embed, attention_mask = attention_mask, labels = labels)
        #的输出结果。它是一个 CausalLMOutputWithCrossAttentions 对象，包含很多字段。.loss 就是语言模型自动计算出来的损失（CrossEntropyLoss），用于训练。
        loss = llm_output.loss
        # 负载均衡损失（从 Megablocks 注册表中取）
        moe_aux_loss = batched_load_balancing_loss(self.moe_config)
        # ✅ 新增：z-loss 计算（聚合所有 router logits）
        z_loss_all = 0.0
        z_loss_coeff = self.moe_config.z_loss_coeff if hasattr(self.moe_config, "z_loss_coeff") else 1e-2
        for block in self.model.model.layers:
           router_logits = block.mlp.router_logits  # 假设你在 MoE 层中保留了 router logits（见提示下方）
           if router_logits is not None:
            z_loss_all += z_loss_func(router_logits, z_loss_coeff)
        
        # 总损失 = LM Loss + 负载均衡损失
        total_loss = loss + moe_aux_loss+ z_loss_all
        # ---------- 7) 按 HuggingFace 规范打包输出 ----------
        #这是 HuggingFace 要求的输出格式，训练时拿 loss，推理时用 logits。
        #CausalLMOutputWithCrossAttentions它是 HuggingFace Transformers 库 里专门为因果语言模型（Causal Language Model，比如 GPT、LLaMA）设计的标准输出格式类。
        return CausalLMOutputWithCrossAttentions(
            loss=total_loss,                               #损失值，只有在训练时传入标签 labels 才会返回
            logits=llm_output.logits,                #模型预测的结果（词概率分布）
            past_key_values=llm_output.past_key_values,#用于加速生成的缓存（可选）
            hidden_states=llm_output.hidden_states,    #所有中间层的输出（可选）
            attentions=llm_output.attentions,          #所有注意力头的权重（可选）
        )
    # =========================================================================
    #  generate() —— 推理 / 生成阶段
    # =========================================================================
    def generate(self, user, item, input_ids):
        #user：用户 ID（整数）,item：物品 ID（整数）,input_ids：prompt 的 token 序列（包含 <USER> 和 <ITEM> 占位符）
        device = input_ids.device
        batch_size = input_ids.size(0)
        for block in self.model.model.layers:
            block.mlp.cluster_index_count  = block.mlp.cluster_index_count + 1
        # ---------- 重复“取嵌入 → 替换占位符”步骤 ----------
        user = torch.tensor(user).contiguous().to(device)
        item = torch.tensor(item).contiguous().to(device)
        
        #user_embed = self.user_embed(user)
        #item_embed = self.item_embed(item)
        #proj_dtype = self.user_proj.weight.dtype
        #user_embed = self.user_embed(user).to(device, dtype=proj_dtype)
        #item_embed = self.item_embed(item).to(device, dtype=proj_dtype)
        user_embed = self.user_embed(user).to(torch.bfloat16)  # 加上 to()
        item_embed = self.item_embed(item).to(torch.bfloat16)  # 加上 to()
        
        user_embeds = self.user_proj(user_embed)
        item_embeds = self.item_proj(item_embed)

        user_index = torch.where(input_ids[0] == 14194)
        item_index = torch.where(input_ids[0] == 37032)

        user_index = user_index[0][0].item()
        item_index = item_index[0][0].item()

        # print(self.model)
        prompt_embeds = self.model.model.embed_tokens(input_ids)

        prompt_embeds[0][item_index] = item_embeds
        prompt_embeds[0][user_index] = user_embeds

        final_embed = prompt_embeds
         # ---------- 调用 Llama-3.generate ----------
        output = self.model.generate(
                inputs_embeds   = final_embed,                      #替换过 <USER> 和 <ITEM> 的向量序列
                do_sample       = True,                             #启用采样生成，提升多样性
                # tokenizer.get_vocab()["<|eot_id|>"] -> 128001     
                pad_token_id    = 128001,                          #padding 位置对应的 token ID
                eos_token_id    = 128001,                          #生成终止符
                max_new_tokens  = 30,                              #最多生成 30 个 token
                temperature     = 0.7,                             #控制生成的随机性，越低越保守
        )
        return output        #最后返回的是生成的 token ID 序列（你可以用 tokenizer.decode() 转为文本

