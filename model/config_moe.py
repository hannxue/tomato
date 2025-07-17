from builtins import object
import torch
class Moe_config(object):
    def __init__(self):
        self.ffn_dim = 256
        self.hidden_dim = 768
        self.num_experts = 12
        self.moe_num_experts = 12
        self.top_k = 2
        self.moe_top_k = 2
        self.num_cluster = 2
        self.moe_loss_weight = 1e-2      # MoE 辅助损失系数
        self.z_loss_coeff = 1e-3
        self.capacity_factor = 1         # 控制 expert 的最大容量
        self.min_capacity = 4                # 控制 expert 的最小容量
        # ⬇️ 新增两行
        self.num_layers                   = 32   # Llama-3 8B has 32 decoder blocks
        self.pipeline_model_parallel_size = 1     # unless you split model by pipeline
        self.num_layers_per_virtual_pipeline_stage = None
        self.moe_lbl_in_fp32 = True  # 或 False，取决于你是否希望 loss 保持 fp32 精度
        self.torch_dtype = torch.bfloat16  # ✅ 添加 dtype 设置

    def __str__(self):
        return (f"Moe_config(ffn_dim={self.ffn_dim}, "
                f"hidden_size={self.hidden_dim}, "
                f"num_cluster={self.num_cluster}, "
                f"top_k={self.top_k})"
                f"moe_top_k={self.moe_top_k})"
                f"moe_loss_weight={self.moe_loss_weight}"
                f"z_loss_coeff={self.z_loss_coeff}"
                f"capacity_factor={self.capacity_factor}, "
                f"min_capacity={self.min_capacity}"
                f"num_layers={self.num_layers}"
                f"moe_num_experts={self.moe_num_experts}"
                f"pipeline_model_parallel_size={self.pipeline_model_parallel_size}"
                f"num_layers_per_virtual_pipeline_stage={self.num_layers_per_virtual_pipeline_stage}"
                f"moe_lbl_in_fp32={self.moe_lbl_in_fp32}"
                f"torch_dtype={self.torch_dtype})")
