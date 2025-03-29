import os
from transformers import TrainingArguments
from llama_factory.arguments import ModelArguments, DataArguments, FinetuningArguments
from llama_factory.trainer import LLaMAFactoryTrainer
from datasets import load_dataset
import wandb

# **1. 初始化 WandB**
wandb.init(project="Qwen2.5-VL-SFT", name="critique-VQA-SFT")

# **2. 设定数据集路径**
DATASET_NAME = "huaXiaKyrie/critique-VQA"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B"

# **3. 加载数据**
dataset = load_dataset(DATASET_NAME)

# **4. 配置数据格式**
def format_data(example):
    return {
        "messages": [
            {"role": "user", "content": example["conversation"]["text"]},
            {"role": "assistant", "content": example["chosen"]["text"]}
        ],
        "images": [example["image"]]  # 直接使用内嵌的 `PIL.Image`
    }

dataset = dataset.map(format_data)

# **5. 训练参数**
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    report_to="wandb",
    fp16=True,
    dataloader_num_workers=4
)

# **6. 额外参数**
model_args = ModelArguments(model_name_or_path=MODEL_NAME)
data_args = DataArguments(dataset=DATASET_NAME, dataset_format="sharegpt")
finetune_args = FinetuningArguments(finetuning_type="lora")

# **7. 训练**
trainer = LLaMAFactoryTrainer(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args,
    finetuning_args=finetune_args,
    dataset=dataset
)

trainer.train()

# **8. 保存模型**
trainer.save_model("./final_model")

# **9. 结束 wandb 记录**
wandb.finish()
