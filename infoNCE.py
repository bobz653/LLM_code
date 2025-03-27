import torch
import torch.nn.functional as F

class ContrastiveModel:
    def __init__(self, encoder, temperature=0.07):
        self.encode = encoder  # 假设 encode 函数可以接收文本并返回其嵌入表示
        self.temperature = temperature

    def forward(self, query, doc, num_negatives_per_query=0):
        """
        :param query: 查询文本或其索引，形状 (B1,)
        :param doc: 文档文本或其索引，形状 (B2,)，其中 B2 = B1 + num_negatives_per_query * B1
        :param num_negatives_per_query: 每个查询构造的人工负例数量，默认为0，意味着仅使用batch内其他样本作为负例
        :return: 计算得到的损失值
        """
        
        # 获取查询和文档的嵌入表示
        q_reps = self.encode(query)  # (B1, d)
        d_reps = self.encode(doc)    # (B2, d)，其中 B2 包含了所有正例和额外的负例
        
        # 计算相似度矩阵
        scores = torch.matmul(q_reps, d_reps.transpose(0, 1)) / self.temperature  # (B1, B2)
        
        # 构建目标标签
        batch_size = q_reps.size(0)
        target = torch.arange(batch_size, device=scores.device, dtype=torch.long)
        
        # 如果有额外的负例，则需要调整target以适应新的scores矩阵大小
        if num_negatives_per_query > 0:
            # 每个查询对应的位置应该是它的索引乘以（1 + num_negatives_per_query）
            target = target * (1 + num_negatives_per_query)
        
        # 计算交叉熵损失.衡量相似度矩阵和标签之间的差距
        loss = F.cross_entropy(scores, target)
        
        return loss
    

# F.cross_entropy 的实现

import torch
import torch.nn.functional as F

# 假设数据
log_probs = torch.tensor([
    [0.1, 0.5, 0.4],
    [0.8, 0.1, 0.1]
], dtype=torch.float32)

target = torch.tensor([1, 0], dtype=torch.long)

# 使用 torch.gather 和 squeeze 选择正确的对数概率  unsquesse的作用是 维度对齐，变成 (B1,1),方便在1这个维度进行gather处理
selected_log_probs = torch.gather(log_probs, 1, target.unsqueeze(1)).squeeze()

print(selected_log_probs)  # 输出: tensor([0.5000, 0.8000])



