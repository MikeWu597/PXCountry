import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer
)
from safetensors.torch import save_model
from PIL import Image

# 配置参数
TEACHER_MODEL_PATH = "/home/vmp/distill/teacher/Qwen2.5-VL-7B-Instruct"
STUDENT_MODEL_PATH = "/home/vmp/distill/student"
DATASET_PATH = "dataset/output.json"
IMAGE_DIR = "dataset/images/"
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5


# 新增视觉适配层
class VisionAdapter(torch.nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        # 添加维度适配层
        self.visual_proj = torch.nn.Linear(
            student.config.encoder_hidden_size,
            teacher.config.encoder_hidden_size
        )

    def forward(self, **inputs):
        outputs = self.student(**inputs)
        # 对齐视觉特征维度
        outputs.encoder_last_hidden_state = self.visual_proj(
            outputs.encoder_last_hidden_state
        )
        return outputs


# 自定义数据集类
class VQADataset(Dataset):
    def __init__(self, data_path, processor, tokenizer):
        with open(data_path) as f:
            self.data = json.load(f)
        self.processor = processor
        self.tokenizer = tokenizer
        if hasattr(self.processor, 'image_processor'):
            self.processor.image_processor.size = {"height": 224, "width": 224}
            self.processor.image_processor.do_resize = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = IMAGE_DIR + item["images"][0]
        image = Image.open(image_path).convert("RGB")

        image_inputs = self.processor.image_processor(
            image,
            return_tensors="pt",
            size=(224, 224),
            do_resize=True,
            do_normalize=True,
            data_format="channels_first"
        )
        print("image_inputs.pixel_values.shape:", image_inputs.pixel_values.shape)  # 新增调试代码

        text_inputs = self.processor.tokenizer(
            item["messages"][0]["content"],
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )
        pixel_values = image_inputs.pixel_values
        # 处理不同维度的情况
        if pixel_values.dim() == 2:  # 灰度图 → 添加通道维度
            pixel_values = pixel_values.unsqueeze(0)  # (1, H, W)
            # 转为 RGB（3 通道）
            pixel_values = pixel_values.expand(3, -1, -1)  # (3, H, W)
        elif pixel_values.dim() != 3:
            raise ValueError(f"Invalid pixel_values dimension: {pixel_values.dim()}")

        # 添加批次维度 → 变为 4D
        pixel_values = pixel_values.unsqueeze(0)  # (1, C, H, W)
        # 对齐视觉特征维度
        if pixel_values.shape[-2:] != torch.Size([224, 224]):
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=(224, 224),
                mode="bilinear"
            )
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze(),
            "attention_mask": text_inputs.attention_mask.squeeze(),
            "labels": self.tokenizer(
                item["messages"][1]["content"],
                return_tensors="pt",
                padding="max_length",
                max_length=32
            ).input_ids.squeeze()
        }


def collate_fn(data):
    pixel_values = torch.stack([d["pixel_values"].squeeze(0) for d in data])  # 假设每个样本的批次维度为1
    input_ids = torch.stack([d["input_ids"] for d in data])
    attention_mask = torch.stack([d["attention_mask"] for d in data])
    labels = torch.stack([d["labels"] for d in data])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 初始化模型和处理器
def load_models():
    # 教师模型（冻结参数）
    teacher = AutoModelForVision2Seq.from_pretrained(
        TEACHER_MODEL_PATH,
        use_safetensors=True
    )
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # 学生模型
    student = AutoModelForVision2Seq.from_pretrained(
        STUDENT_MODEL_PATH,
        use_safetensors=True
    )
    # 在加载模型后添加适配层
    # student = VisionAdapter(student, teacher)
    # 共享处理器
    processor = AutoProcessor.from_pretrained(TEACHER_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH)

    return teacher, student, processor, tokenizer


# 自定义训练器实现知识蒸馏
class DistillationTrainer(Trainer):
    def __init__(self, teacher=None, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher  # 显式接收教师模型

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # 获取图像张量和文本输入
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        # 学生模型前向传播（使用完整的模型接口）
        student_outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # 教师模型前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

        # 计算损失
        loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
        return (loss, student_outputs) if return_outputs else loss

    def distillation_loss(self, student, teacher, labels):
        # 视觉特征蒸馏（使用视觉编码器的输出）
        vision_loss = torch.nn.functional.mse_loss(
            student_visual.encoder_last_hidden_state,  # 假设 student_visual 是视觉编码器的输出
            teacher_visual.encoder_last_hidden_state.detach()
        )

        # 文本特征蒸馏
        text_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student.logits, dim=-1),
            torch.nn.functional.softmax(teacher.logits.detach(), dim=-1),
            reduction="batchmean"
        )

        # 任务损失
        task_loss = student.loss

        return 0.4 * vision_loss + 0.4 * text_loss + 0.2 * task_loss


# 主训练流程
def main():
    # 加载模型
    teacher, student, processor, tokenizer = load_models()

    # 准备数据集
    dataset = VQADataset(DATASET_PATH, processor, tokenizer)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练配置
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_strategy="no",
        remove_unused_columns=False
    )

    # 创建训练器
    trainer = DistillationTrainer(
        model=student,
        teacher=teacher,  # 自定义参数传递
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn
    )

    # 开始训练
    trainer.train()

    # 保存模型
    student.save_pretrained("./distilled_model", safe_serialization=True)
    processor.save_pretrained("./distilled_model")
    tokenizer.save_pretrained("./distilled_model")


if __name__ == "__main__":
    main()
