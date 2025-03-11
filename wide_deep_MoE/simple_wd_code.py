import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

class WideAndDeepWithBERTandMoE(tf.keras.Model):
    def __init__(self, num_experts=3, output_dims=1):
        super(WideAndDeepWithBERTandMoE, self).__init__()
        # BERT模型作为deep部分
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        # Wide部分: 简单的线性层
        self.wide = tf.keras.layers.Dense(output_dims)
        # 专家网络
        self.experts = [tf.keras.layers.Dense(output_dims) for _ in range(num_experts)]
        # 路由器网络
        self.router = tf.keras.layers.Dense(num_experts, activation='softmax')
        # 任务头：针对不同任务的输出层
        self.task_heads = [tf.keras.layers.Dense(output_dims, activation='sigmoid') for _ in range(2)]  # CTR和CVR
        
    def call(self, inputs):
        wide_inputs, deep_inputs = inputs['wide'], inputs['deep']
        # 处理文本输入
        deep_outputs = self.bert(deep_inputs)[1]  # 获取pooled输出
        # Wide部分
        wide_output = self.wide(wide_inputs)
        # 合并wide和deep的结果
        combined_output = tf.keras.layers.concatenate([wide_output, deep_outputs])
        
        # 计算路由权重
        router_weights = self.router(combined_output)  # Shape: (batch_size, num_experts)
        
        # 对每个专家进行计算，并加权求和
        expert_outputs = [expert(combined_output) for expert in self.experts]
        expert_outputs_stack = tf.stack(expert_outputs, axis=-1)  # Shape: (batch_size, output_dims, num_experts)
        weighted_expert_outputs = tf.reduce_sum(expert_outputs_stack * tf.expand_dims(router_weights, axis=1), axis=-1)
        
        # 通过任务头得到最终输出
        task_outputs = [head(weighted_expert_outputs) for head in self.task_heads]
        
        return task_outputs

# 创建模型实例
model = WideAndDeepWithBERTandMoE(num_experts=3)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={'output_1': 'binary_crossentropy', 'output_2': 'binary_crossentropy'},
              metrics=['accuracy'])

# 准备数据
# 这里假设您已经有了训练和测试数据集
# train_dataset, test_dataset = prepare_datasets()

# 训练模型
# model.fit(train_dataset, epochs=5, validation_data=test_dataset)

print("模型构建完成")
