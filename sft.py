import json
import os
from PIL import Image
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
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
image_processor = AutoImageProcessor.from_pretrained(BASE_MODEL_PATH)

# 加载并处理数据集
with open(DATASET_PATH) as f:
    raw_data = json.load(f)


def process_example(example):
    # 处理图像
    img_path = os.path.join(IMAGE_DIR, example["images"][0])
    image = Image.open(img_path).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values[0]

    # 处理对话格式
    messages = example["messages"]
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": tokenized[0],
        "labels": tokenized[0].clone()
    }


# 创建HuggingFace数据集
dataset = Dataset.from_list(raw_data)
dataset = dataset.map(
    process_example,
    remove_columns=["messages", "images"]
)
# 在训练前添加数据验证
sample = dataset[0]
print("Pixel values type:", type(sample["pixel_values"]))  # 应该显示torch.Tensor
print("Input ids type:", type(sample["input_ids"]))       # 应该显示torch.Tensor


# 自定义数据整理函数
def collate_fn(batch):
    # 处理pixel_values
    pixel_values = []
    for x in batch:
        pv = x["pixel_values"]
        if not isinstance(pv, torch.Tensor):
            pv = torch.tensor(pv)
        pixel_values.append(pv)

    # 处理文本序列
    input_ids = [x["input_ids"] for x in batch]
    labels = [x["labels"] for x in batch]

    return {
        "pixel_values": torch.stack(pixel_values),
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )
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
