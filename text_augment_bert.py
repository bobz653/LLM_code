# 希望基于 BERT 或 T5 实现更高级的文本增强方式（比如同义替换、回译、插入干扰词等），也可以告诉我，请提供对应的方案。 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例语料
sentences = [
    "我喜欢深度学习",
    "机器学习很有意思",
    "生成对抗网络很强大",
    "我要去留学读硕士"
]

# 构建字符字典
chars = set(''.join(sentences))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = list(chars)

VOCAB_SIZE = len(chars)
MAX_LEN = max(len(s) for s in sentences)

# 将字符串转为张量
def text_to_tensor(text):
    indices = [char_to_idx[c] for c in text]
    return torch.tensor(indices, dtype=torch.long)

# 填充序列
def collate_fn(batch):
    batch = pad_sequence(batch, padding_value=0, batch_first=True)
    return batch

# 转换为 tensor 数据集
dataset = [text_to_tensor(s) for s in sentences]
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, vocab_size, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len
        self.latent_dim = latent_dim

    def forward(self, z):
        h0 = torch.zeros(1, z.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(1, z.size(0), self.lstm.hidden_size).to(device)
        x = torch.randn(z.size(0), self.max_len, self.latent_dim).to(device)
        x, _ = self.lstm(x, (h0, c0))
        logits = self.linear(x)
        return logits

# 判别器
class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(VOCAB_SIZE, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.linear(out.mean(dim=1))
        return logits

# 超参数
latent_dim = 32
hidden_dim = 64
lr = 0.001
epochs = 200
batch_size = 2

# 初始化模型
generator = Generator(latent_dim, hidden_dim, VOCAB_SIZE, MAX_LEN).to(device)
discriminator = Discriminator(hidden_dim).to(device)

# 损失函数和优化器
adversarial_loss = nn.BCEWithLogitsLoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# 训练循环
for epoch in range(epochs):
    for real_data in data_loader:
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # one-hot 编码
        real_data_onehot = torch.zeros(batch_size, MAX_LEN, VOCAB_SIZE).to(device)
        real_data_onehot.scatter_(2, real_data.unsqueeze(-1), 1)

        # ---------------------
        #  训练判别器 D
        # ---------------------
        d_optimizer.zero_grad()

        # 真样本损失
        real_logits = discriminator(real_data_onehot)
        d_real_loss = adversarial_loss(real_logits, real_labels)

        # 生成假样本
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_logits = generator(z)
        fake_samples = torch.multinomial(F.softmax(fake_logits.view(-1, VOCAB_SIZE), dim=1), 1).view(batch_size, -1)

        # 假样本损失
        fake_data_onehot = torch.zeros(batch_size, MAX_LEN, VOCAB_SIZE).to(device)
        fake_data_onehot.scatter_(2, fake_samples.unsqueeze(-1), 1)
        fake_logits = discriminator(fake_data_onehot.detach())
        d_fake_loss = adversarial_loss(fake_logits, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # ---------------------
        #  训练生成器 G
        # ---------------------
        g_optimizer.zero_grad()
        fake_logits = discriminator(fake_data_onehot)
        g_loss = adversarial_loss(fake_logits, real_labels)
        g_loss.backward()
        g_optimizer.step()

    # 每隔一定轮次打印结果
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# 生成扰动文本
def generate_text():
    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        output = generator(z)
        _, indices = torch.max(output, dim=2)
        indices = indices[0].cpu().numpy()
        text = ''.join([idx_to_char[i] for i in indices])
        return text

# 扰动增强示例
print("\n生成扰动后的文本示例：")
for _ in range(5):
    print(generate_text())
