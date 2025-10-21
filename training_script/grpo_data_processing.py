import json
import os
from sklearn.model_selection import train_test_split

# 输入和输出文件路径
input_file = "geo170k_train_data.jsonl"
train_output_file = "geo170k_train.json"
test_output_file = "geo170k_test.json"

# 基础图像路径前缀
base_image_path = "XXXXX"

# 尝试读取整个文件作为一个 JSON 列表
with open(input_file, 'r') as f:
    try:
        data = json.load(f)
        print(f"文件包含一个包含 {len(data)} 个元素的列表")

        converted_data = []
        for item in data:
            if not isinstance(item, dict):
                print(f"警告: 列表中的元素不是字典")
                continue

            if "problem" not in item or "image_path" not in item or "solution" not in item:
                print(f"警告: 元素缺少必要的键")
                continue

            # 构建转换后的对象
            converted_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": item["problem"] + "<image>"
                    }
                ],
                "images": [
                    os.path.join(base_image_path, item["image_path"])
                ],
                "solution": item["solution"]
            }

            converted_data.append(converted_item)

    except json.JSONDecodeError as e:
        print(f"错误: 文件不是有效的 JSON: {e}")
        exit()

# 其余代码保持不变...
print(f"成功处理 {len(converted_data)} 条有效数据")

if len(converted_data) == 0:
    print("没有有效数据可处理，退出程序")
    exit()

# 分割数据集
train_data, test_data = train_test_split(
    converted_data,
    test_size=0.05,
    random_state=42  # 设置随机种子以确保结果可重现
)

# 将训练集数据写入文件
with open(train_output_file, 'w') as f:
    json.dump(train_data, f, indent=2)

# 将测试集数据写入文件
with open(test_output_file, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"转换完成！共处理 {len(converted_data)} 条数据。")
print(f"训练集: {len(train_data)} 条数据")
print(f"测试集: {len(test_data)} 条数据")
print(f"训练集已保存到 {train_output_file}")
print(f"测试集已保存到 {test_output_file}")
