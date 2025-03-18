import json
import os
import math
from PIL import Image
import numpy as np
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import safetensors
# 配置参数
BASE_MODEL_PATH = "/home/vmp/distill/student"  # 原始模型路径
DATASET_PATH = "dataset/output.json"  # 数据集路径
IMAGE_DIR = "dataset/images/"  # 图片存储目录
OUTPUT_DIR = "res"  # 输出目录

# 加载模型和处理器
model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True  # 必须开启远程代码支持
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
image_processor = AutoImageProcessor.from_pretrained(BASE_MODEL_PATH)

# 加载并处理数据集
with open(DATASET_PATH) as f:
    raw_data = json.load(f)


def process_example(example):
    img_path = os.path.join(IMAGE_DIR, example["images"][0])
    image = Image.open(img_path).convert("RGB")

    # 获取基础视觉特征
    visual_inputs = image_processor(
        image,
        return_tensors="pt"
    )

    # 自动计算grid_thw（核心修改）
    h, w = image.size
    patch_size = model.config.vision_config.patch_size  # 通常是14
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    grid_thw = (1, num_patches_h, num_patches_w)  # 时间维度固定为1

    # 处理对话格式
    messages = example["messages"]
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    return {
        "pixel_values": visual_inputs.pixel_values[0].numpy(),
        "input_ids": tokenized[0].numpy(),
        "labels": tokenized[0].clone().numpy(),
        "image_h": h,  # 存储原始尺寸
        "image_w": w
    }


# 创建HuggingFace数据集
dataset = Dataset.from_list(raw_data)


# 自定义数据整理函数
def collate_fn(batch):
    # 转换所有字段为Tensor
    def to_tensor(data):
        if isinstance(data, np.ndarray):
            return torch.tensor(data)
        return data

    # 处理所有字段
    processed_batch = []
    for x in batch:
        processed_batch.append({
            "pixel_values": to_tensor(x["pixel_values"]),
            "input_ids": to_tensor(x["input_ids"]),
            "labels": to_tensor(x["labels"]),
            "image_h": x["image_h"],
            "image_w": x["image_w"]
        })

    # 提取并处理pixel_values
    pixel_values = torch.stack([x["pixel_values"] for x in processed_batch])

    # 动态计算grid_thw
    batch_h = [x["image_h"] for x in processed_batch]
    batch_w = [x["image_w"] for x in processed_batch]
    patch_size = model.config.vision_config.patch_size
    grid_thw = [
        (1, math.ceil(h / patch_size), math.ceil(w / patch_size))
        for h, w in zip(batch_h, batch_w)
    ]

    # 处理文本序列（关键修改）
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [x["input_ids"] for x in processed_batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [x["labels"] for x in processed_batch],
        batch_first=True,
        padding_value=-100
    )

    return {
        "pixel_values": pixel_values,
        "grid_thw": grid_thw,
        "input_ids": input_ids,
        "labels": labels
    }


# 配置训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    remove_unused_columns=False
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

# 开始训练
trainer.train()

# 保存最终模型
model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True
)

print(f"Distilled model saved to {OUTPUT_DIR}")
