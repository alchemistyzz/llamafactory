import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import io

# **文件路径**
parquet_path = "/home/v-yizhezhang/data/geo3k/test-00000-of-00001.parquet"
image_output_folder = "/home/v-yizhezhang/data/geo3k/test_images"
output_json_file = "/home/v-yizhezhang/code/LLaMA-Factory/data/geo3k_llamafactory.json"

# **创建存储图片文件夹**
os.makedirs(image_output_folder, exist_ok=True)

# **读取 parquet 文件**
df = pd.read_parquet(parquet_path)

# **检查数据列**
print(f"🔍 数据列名: {df.columns.tolist()}")

# **存储转换后的数据**
data_list = []

# **遍历数据集**
for index, row in df.iterrows():
    problem_text = row["problem"]  # **题目**
    answer_text = row["answer"]  # **答案**
    choices_list = row["choices"]  # **选项**
    image_data_list = row["images"]  # **图片数据（列表）**

    # **处理图片**
    image_paths = []
    for i, image_data in enumerate(image_data_list):
        if isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]  # **获取二进制数据**
            image_filename = f"{row['id']}_{i}.png"  # **生成图片文件名**
            image_path = os.path.join(image_output_folder, image_filename)

            # **存储 PNG 图片**
            image = Image.open(io.BytesIO(image_bytes))
            image.save(image_path, format="PNG")
            image_paths.append(image_path)

    # **确保 `<image>` 只出现一次**
    if "<image>" not in problem_text:
        problem_text = "<image>\n" + problem_text  # **如果没有 `<image>`，则添加**

    # **构造符合 LLaMA-Factory 格式的 `messages`**
    messages = [
        {
            "content": f"{problem_text}\nchoices are {', '.join(choices_list)}\nPlease give the correct option (A/B/C/D).",
            "role": "user",
        },
        {
            "content": f"The correct answer is {answer_text}.",
            "role": "assistant",
        },
    ]

    # **存储最终数据**
    data = {
        "messages": messages,
        "images": image_paths if image_paths else []
    }

    data_list.append(data)

# **保存 JSON 文件**
with open(output_json_file, "w", encoding="utf-8") as json_file:
    json.dump(data_list, json_file, indent=4, ensure_ascii=False)

print(f"✅ 数据成功转换并保存至: {output_json_file}")
print(f"📂 图片已保存至文件夹: {image_output_folder}")
