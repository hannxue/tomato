
prompt_template = {
    'instruction': 'You are a professional explainer, Your assignment involves providing users with a detailed explanation regarding a specific item.',
     # 系统设定，告诉语言模型：你是一个解释专家，要根据用户-物品信息生成推荐解释
    'input'      : 'The user has a {} experience with the item, the item has {} features, please provide an explanation for recommending {} to {}.'
    # 输入提示模板，用于动态填入情感、特征、物品占位符、用户占位符（后续会替换 ITEM/USER）用户对该项目有｛｝体验，该项目具有｛｝个功能，请提供向｛｝推荐｛｝的解释。
}
# ---------------------- prompt 构造与编码处理类 ----------------------
class Prompt_Process:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer        # HuggingFace 的 tokenizer
        self.MAX_LENGTH = max_seq_length  # 最大序列长度，用于截断控制
    def __call__(self,examples):
        input_ids, attention_mask, labels = [], [], []  # 初始化输出字段
        # 根据评分判断情绪倾向（大于等于 3 为 positive，小于 3 为 negative）
        mood = 'positive' if examples['rating'] >= 3 else 'negative' 
        # 取出特征、用户 ID、物品 ID
        feature = examples['feature']
        user = examples['user']
        item = examples['item']
        # 构造自然语言提示：把情绪、特征等填入 prompt
        #故意设计成占位符字符串，为了后续能用 嵌入替换 的机制注入协同过滤信息（user/item embedding），而不是简单地把 ID 打印出来。
        inputs = prompt_template['input'].format(mood, feature, 'ITEM', 'USER')#当前数据集中，你没有 item_name 和 user_name 字段，只有 item 和 user 是 ID（整数），直接填进去意义不大，反而干扰了 prompt 的一致性。
        # 构造系统 + 用户输入部分（不加 <bos>/<eos> 特殊符号）
        instruction = self.tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt_template['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{inputs}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  
        #<|begin_of_text|>
        #<|start_header_id|>system<|end_header_id|>
        #你是一个专业的推荐解释员……
        #<|eot_id|>
        #<|start_header_id|>user<|end_header_id|>
        #这个用户对商品有积极体验……
        #<|eot_id|>
        #<|start_header_id|>assistant<|end_header_id|>

        # 构造模型应答部分（即 ground-truth 推荐解释文字），把你原始数据中提供的推荐解释文本（text 字段）转为 token。
        output = self.tokenizer(f"{examples['text']}<|eot_id|>", add_special_tokens=False)
        
        # 拼接整体输入 token 序列（instruction + output + pad）[系统+用户指令内容 token] + [ground-truth 解释内容 token] + [padding token]
        #"input_ids" 是 HuggingFace Tokenizer 分词器在对文本进行编码时生成的 Token ID 列表，它表示输入文本在词表中的索引序列。
        input_ids = instruction["input_ids"] + output["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + output["attention_mask"] + [1]  # pad 部分也加上 mask = 1,	构造 attention mask，指示模型哪些位置是有效的输入。
        # instruction + input mask -100
        # 构造训练用标签（instruction 部分的 token 不需要训练 → 设置为 -100 以跳过）用 -100 来屏蔽这一段的梯度计算,把 instruction 部分设为 -100 表示不训练那部分（即不需要模型去“记住问题”，只要学会生成答案
        labels = [-100] * len(instruction["input_ids"]) + output["input_ids"] + [self.tokenizer.pad_token_id]  

        # truncation # 如果总长度超过最大限制，进行截断处理
        if len(input_ids) > self.MAX_LENGTH:  
            input_ids = input_ids[:self.MAX_LENGTH]
            attention_mask = attention_mask[:self.MAX_LENGTH]
            labels = labels[:self.MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }