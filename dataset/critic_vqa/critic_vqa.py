import os
from datasets import load_dataset

# 设定数据集名称
DATASET_NAME = "huaXiaKyrie/critique-VQA"

# 加载数据集
dataset = load_dataset(DATASET_NAME)

# 设定输出路径
output_path = "data/critique-VQA-SFT"
os.makedirs(output_path, exist_ok=True)

# # 只筛选前 100 条数据
# test_size = 5

for split in dataset.keys():
    # print(f"正在处理 {split} 数据集，仅提取前 {test_size} 条数据...")

    # 先筛选前100条数据
    # sampled_data = dataset[split].select(range(min(test_size, len(dataset[split]))))

    # 处理数据格式
    def format_data(example):
        messages = []
        if "conversation" in example and isinstance(example["conversation"], dict):
            user_message = example["conversation"].get("text", "")
            messages.append({"role": "user", "content": user_message})

        if "chosen" in example and isinstance(example["chosen"], dict):
            assistant_response = example["chosen"].get("text", "")
            messages.append({"role": "assistant", "content": assistant_response})

        images = [example["image"]] if "image" in example else []

        return {
            "messages": messages,
            "images": images  # 直接保留 PIL.Image
        }

    formatted_data = dataset[split].map(format_data, num_proc=128)

    # 确保 `split` 目录存在
    split_output_path = os.path.join(output_path, split)
    os.makedirs(split_output_path, exist_ok=True)

    # 存储为 Parquet（分片格式）
    parquet_file_path = os.path.join(split_output_path, f"{split}.parquet")
    formatted_data.to_parquet(parquet_file_path)

    # print(f"{split} 数据已成功转换并存储至 `{parquet_file_path}`（前 {test_size} 条样本）")

print("所有数据处理完成 ✅")
