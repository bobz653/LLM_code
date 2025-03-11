# tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_with_bert(text):
    tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
    return tokens

# 示例
print(tokenize_with_bert("Hello, world! This is a test."))

# BertForTokenClassification

from transformers import BertForTokenClassification, Trainer, TrainingArguments

# 假设已准备好数据集
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels) # num_labels应根据实际标签数量设置

training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=64, warmup_steps=500, weight_decay=0.01, logging_dir='./logs',)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)

trainer.train()

# BertForSequenceClassification

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes) # num_classes是分类的数量

training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=64, warmup_steps=500, weight_decay=0.01, logging_dir='./logs',)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)

trainer.train()
