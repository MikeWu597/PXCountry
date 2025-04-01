
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoImageProcessor,
    AutoTokenizer,
    default_data_collator
)
from datasets import load_dataset
import json
import os

# 配置参数
BASE_IMAGE_DIR = 'dataset/images/'  # 图片存储目录
MODEL_PATH = "/home/vmp/distill/student"  # 原始模型路径
DATASET_PATH = "dataset/output.json"  # 数据集路径
SAVE_PATH = 'res'  # 模型保存路径

# 硬件配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和处理器
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
).to(device)
image_processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    # 处理grid_thw问题（假设模型需要特定网格参数）
    size=(256,256),  # 根据实际情况调整
    do_resize=True
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    pad_token='<pad>'  # 确保设置填充token
)


# 自定义数据集类
class StampDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 处理图像
        img_path = os.path.join(BASE_IMAGE_DIR, item['images'][0])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((256,256))

        # 使用图像处理器处理并确保tensor格式
        pixel_values = image_processor(
            image,
            return_tensors="pt",
            # 处理grid尺寸对齐问题
            do_resize=True,
            size=(256,256),
            do_normalize=True
        ).pixel_values.squeeze(0)  # 移除批次维度

        # 处理文本
        messages = item['messages']
        question = messages[0]['content']
        answer = messages[1]['content']

        # 编码输入文本（处理<image>特殊标记）
        inputs = tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        )

        # 编码标签（处理填充对齐）
        labels = tokenizer(
            answer,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32
        ).input_ids

        # 掩码处理（将填充部分设为-100）
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'pixel_values': pixel_values,
            'labels': labels.squeeze(0)
        }


# 初始化数据集和数据加载器
dataset = StampDataset(DATASET_PATH)


def collate_fn(batch):
    # 处理不同尺寸问题（确保图像尺寸统一）
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

# 训练配置
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

# 训练循环
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        # 数据转移到设备
        inputs = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        outputs = model(**inputs)
        loss = outputs.loss

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Avg Loss: {total_loss / len(train_loader):.4f}")

# 保存蒸馏后的模型
model.save_pretrained(
    SAVE_PATH,
    safe_serialization=True,
    variant='fp16'  # 可选，节省存储空间
)

print(f"模型已成功保存至 {SAVE_PATH}")
