import requests
import base64
import os
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import argparse
import pandas as pd
import concurrent.futures
from multiprocessing import Manager, Lock
import time
import re
import sys
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
# ========================
# 1. 辅助函数
# ========================
def safe_parse(text):
    """安全解析模型输出中的答案"""
    try:
        # 尝试匹配<answer>标签
        answer_match = re.search(r"<answer>\s*([\d.]+)", text, re.IGNORECASE)
        if answer_match:
            return [float(answer_match.group(1))]

        # 尝试匹配文本中的数字
        numbers = re.findall(r"\d+\.?\d*", text)
        return [float(numbers[-1])] if numbers else None
    except:
        return None

def safe_verify(pred, truth, tolerance=1e-3):
    """验证预测答案是否正确"""
    if not pred or not truth:
        return 0.0
    return 1.0 if abs(pred[0] - truth[0]) < tolerance else 0.0

# ========================
# 2. 视觉语言消息客户端
# ========================
class VLMessageClient:
    def __init__(self, api_url, model_name):
        self.api_url = api_url
        self.model_name = model_name
        self.session = requests.Session()

    def _encode_image(self, image_path):
        """编码图像为base64"""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def build_messages(self, item, image_root):
        """构建消息"""
        # 1. 获取图像路径
        image_path = os.path.join(image_root, item['image_path'].lstrip('./'))

        # 2. 构建消息（按照要求修改）
        return [
            {
                "role": "system",
                "content": (
                    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"}},
                    {
                        "type": "text",
                        "text": f"{item['question']}"
                    }
                ]
            }
        ]

    def process_item(self, item, image_root, output_file, error_file, total_counter, correct_counter, lock):
        """处理单个项目"""
        max_retries = 5
        attempt = 0
        result = None

        while attempt < max_retries:
            try:
                attempt += 1
                messages = self.build_messages(item, image_root)

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "top_p": 1,
                    "repetition_penalty": 1.00
                }

                response = self.session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=100 + attempt * 5
                )
                response.raise_for_status()

                output = response.json()["choices"][0]["message"]["content"]

                # 解析和验证答案
                gt = safe_parse(item["ground_truth"])
                pred = safe_parse(output)
                is_correct = bool(pred and gt and safe_verify(pred, gt))

                with lock:
                    total_counter.value += 1
                    correct_counter.value += int(is_correct)
                    current_accuracy = float(correct_counter.value) / float(total_counter.value) if total_counter.value > 0 else 0.0

                # 构建成功结果
                result = {
                    "question": str(item["question"]),
                    "image_path": str(item["image_path"]),
                    "model_output": str(output),
                    "extracted_answer": str(pred[0]) if pred else None,
                    "ground_truth": str(item["ground_truth"]),
                    "is_correct": bool(is_correct),
                    "current_correct": int(correct_counter.value),
                    "current_total": int(total_counter.value),
                    "current_accuracy": float(current_accuracy),
                    "attempt": int(attempt),
                    "success": bool(True),
                    "model": self.model_name
                }

                # 写入成功文件
                with lock:
                    try:
                        with open(output_file, "a") as f:
                            json.dump(result, f, ensure_ascii=False, default=str)
                            f.write("\n")
                            f.flush()
                    except Exception as e:
                        print(f"写入成功文件失败: {str(e)}")

                return True

            except Exception as e:
                if attempt == max_retries:
                    # 构建失败结果
                    error_data = {
                        "question": str(item["question"]),
                        "image_path": str(item["image_path"]),
                        "error": str(e),
                        "attempt": int(attempt),
                        "success": bool(False)
                    }
                    # 写入错误文件
                    with lock:
                        try:
                            with open(error_file, "a") as f:
                                json.dump(error_data, f, ensure_ascii=False, default=str)
                                f.write("\n")
                                f.flush()
                        except Exception as e:
                            print(f"写入错误文件失败: {str(e)}")
                    return False
                else:
                    time.sleep(min(2 ** attempt, 10))

        return False

# ========================
# 3. 主函数
# ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://127.0.0.1:8000")
    parser.add_argument("--model_name", required=True, help="模型名称")
    parser.add_argument("--prompt_path", required=True, help="测试集路径")
    parser.add_argument("--image_root", default="../", help="图像根目录")
    parser.add_argument("--output_path", required=True, help="输出文件路径")
    parser.add_argument("--max_workers", type=int, default=3, help="最大工作进程数")
    args = parser.parse_args()

    # 设置错误文件路径
    error_output_path = os.path.splitext(args.output_path)[0] + "_errors.jsonl"

    # 1. 加载测试数据
    test_data_all = pd.read_json(args.prompt_path, lines=True).to_dict("records")
    total_samples = len(test_data_all)
    print(f"测试集总数: {total_samples}")
    print(f"使用模型: {args.model_name}")

    # 2. 恢复已成功处理的数据（只恢复success为true的记录）
    processed_success = set()  # 成功处理的数据
    recovered_total = 0
    recovered_correct = 0
    valid_records = []  # 有效的成功记录

    # 检查输出文件是否存在
    output_file_exists = os.path.exists(args.output_path)

    # 恢复成功记录并清理输出文件
    if output_file_exists:
        print("检查输出文件...")
        with open(args.output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("success", False):
                        processed_success.add(data["image_path"])
                        recovered_total += 1
                        if data.get("is_correct", False):
                            recovered_correct += 1
                        valid_records.append(data)
                except json.JSONDecodeError:
                    continue

        # 重写输出文件，只保留成功记录
        if valid_records:
            print(f"找到 {len(valid_records)} 条成功记录")
            with open(args.output_path, "w") as f:
                for record in valid_records:
                    json.dump(record, f, ensure_ascii=False, default=str)
                    f.write("\n")
        else:
            print("输出文件中没有有效记录，创建空文件")
            open(args.output_path, 'w').close()
    else:
        # 创建输出文件
        print("输出文件不存在，创建新文件")
        open(args.output_path, 'w').close()

    # 安全打印恢复统计信息
    print(f"已成功恢复记录数: {recovered_total} ({recovered_total/total_samples:.2%})")
    if recovered_total > 0:
        print(f"已恢复正确记录数: {recovered_correct} ({recovered_correct/recovered_total:.2%})")
    else:
        print(f"已恢复正确记录数: 0 (N/A)")

    # 3. 确定剩余待处理数据（所有未成功的数据）
    remaining_data = []
    for item in test_data_all:
        img_path = item["image_path"]
        if img_path not in processed_success:
            remaining_data.append(item)

    print(f"剩余待处理记录数: {len(remaining_data)}")

    # 4. 处理剩余数据
    if remaining_data:
        print(f"开始处理剩余记录，使用 {args.max_workers} 个工作进程...")

        # 确保错误文件存在
        if not os.path.exists(error_output_path):
            open(error_output_path, 'w').close()

        with Manager() as manager:
            total_counter = manager.Value('i', recovered_total)
            correct_counter = manager.Value('i', recovered_correct)
            lock = manager.Lock()

            with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                client = VLMessageClient(args.api_url, args.model_name)

                for item in remaining_data:
                    futures.append(
                        executor.submit(
                            client.process_item,
                            item=item,
                            image_root=args.image_root,
                            output_file=args.output_path,
                            error_file=error_output_path,
                            total_counter=total_counter,
                            correct_counter=correct_counter,
                            lock=lock
                        )
                    )

                # 5. 进度条
                with tqdm(total=len(remaining_data), desc="处理进度") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"处理异常: {str(e)}")
                        finally:
                            pbar.update(1)
                            current_total = total_counter.value
                            current_correct = correct_counter.value
                            processed_info = f"{current_total}/{total_samples}"

                            # 避免除零错误
                            if current_total > 0:
                                accuracy_info = f"{current_correct/current_total:.2%}"
                            else:
                                accuracy_info = "N/A"

                            pbar.set_postfix({
                                "当前正确": current_correct,
                                "当前总数": current_total,
                                "准确率": accuracy_info,
                                "已处理": processed_info
                            })

    # 6. 最终统计
    # 统计成功文件
    success_count = 0
    correct_count = 0
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("success", False):
                        success_count += 1
                        if data.get("is_correct", False):
                            correct_count += 1
                except:
                    continue

    # 统计错误文件
    error_count = 0
    if os.path.exists(error_output_path):
        with open(error_output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if not data.get("success", True):
                        error_count += 1
                except:
                    continue

    total_processed = success_count + error_count

    # 避免除零错误
    if success_count > 0:
        final_accuracy = correct_count / success_count
    else:
        final_accuracy = 0

    print("\n最终统计:")
    print(f"测试集总数: {total_samples}")
    print(f"成功处理记录数: {success_count} ({success_count/total_samples:.2%})")
    print(f"失败处理记录数: {error_count} ({error_count/total_samples:.2%})")

    if success_count > 0:
        print(f"正确结果数: {correct_count} ({final_accuracy:.2%})")
    else:
        print(f"正确结果数: 0 (N/A)")

    print(f"总处理记录数: {total_processed} (应与测试集总数一致: {'是' if total_processed == total_samples else '否'})")

    # 保存统计文件
    stats_path = os.path.splitext(args.output_path)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total_samples": total_samples,
            "model_name": args.model_name,
            "successful_inferences": success_count,
            "failed_inferences": error_count,
            "correct_results": correct_count,
            "accuracy": final_accuracy,
            "output_file": args.output_path,
            "error_file": error_output_path
        }, f, indent=4)

    print(f"统计信息已保存到: {stats_path}")

if __name__ == "__main__":
    main()
