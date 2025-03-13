import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import ViTFeatureExtractor, ViTModel, BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from PIL import Image
import requests

class ImageTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, feature_extractor, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_url = row['image_url']
        text = row['text']

        # 下载图片
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        
        # 图像特征提取
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        
        # 文本编码
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'pixel_values': pixel_values.squeeze(),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }

class ImageTextModel(nn.Module):
    def __init__(self, vit_model, bert_model, freeze_vit=True):
        super(ImageTextModel, self).__init__()
        self.vit = vit_model
        self.bert = bert_model
        self.classifier = nn.Linear(vit_model.config.hidden_size + bert_model.config.hidden_size, 2)  # 输出类别数

        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        visual_features = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        combined_features = torch.cat((visual_features, text_features), dim=1)
        output = self.classifier(combined_features)
        return output


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values, input_ids, attention_mask)
        loss = loss_fn(outputs, batch['labels'].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(pixel_values, input_ids, attention_mask)
            loss = loss_fn(outputs, batch['labels'].to(device))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练模型
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    
    # 构建模型
    model = ImageTextModel(vit_model, bert_model).to(device)
    
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 数据加载
    df = pd.read_csv('your_dataset.csv')  # 请替换为实际的数据路径
    dataset = ImageTextDataset(df, BertTokenizer.from_pretrained('bert-base-chinese'), ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k'))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 训练
    for epoch in range(3):  # 调整epoch数量
        train_loss = train_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')

if __name__ == "__main__":
    main()
