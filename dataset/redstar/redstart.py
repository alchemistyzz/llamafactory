import json
import os

# 指定数据源和本地图片文件夹
local_image_folder = '/home/v-yizhezhang/data/redstar'
jsonl_file = '/home/v-yizhezhang/code/math_level/geo_multimodal.jsonl'
output_json_file = '/home/v-yizhezhang/code/LLaMA-Factory/data/redstar.json'

data_list = []

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        
        problem = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        image_filename = item['image']
        image_path = os.path.join(local_image_folder, image_filename)
        
        # 如果问题中没有<image>，则在开头添加
        if "<image>" not in problem:
            problem = "<image>" + problem
        
        # 构造符合目标格式的数据
        data = {
            "messages": [
                {
                    "content": problem,
                    "role": "user"
                },
                {
                    "content": answer,
                    "role": "assistant"
                }
            ],
            "images": [image_path]
        }
        
        data_list.append(data)

# 将数据保存为 JSON 文件
with open(output_json_file, 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, indent=4, ensure_ascii=False)

print("Data successfully converted and saved in the desired format.")