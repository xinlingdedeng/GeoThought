#!/usr/bin/env python3
"""
Parquet 数据解析脚本
用于解析包含图像二进制数据和嵌套字典的 Parquet 文件
并生成 JSONL 格式的输出
"""

import pandas as pd
import json
from PIL import Image
import io
import os
import argparse
from pathlib import Path
import base64
import numpy as np

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='解析包含图像和嵌套数据的Parquet文件并生成JSONL')
    parser.add_argument('input_file', type=str, help='输入的Parquet文件路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录路径，默认: ./output')
    parser.add_argument('--image_dir', type=str, default='images',
                       help='图像存储子目录，默认: images')
    parser.add_argument('--max_rows', type=int, default=None,
                       help='最大处理行数（用于测试），默认: 处理所有行')
    parser.add_argument('--embed_images', action='store_true',
                       help='将图像嵌入JSONL中（Base64编码），而不是保存为文件')
    return parser.parse_args()

def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def process_image_data(img_data, index, save_dir=None, embed=False):
    """
    处理图像数据：保存为文件或编码为Base64

    Args:
        img_data: 图像数据字典
        index: 索引，用于命名
        save_dir: 图片保存的目录（如果不嵌入）
        embed: 是否将图像嵌入JSONL中

    Returns:
        dict: 包含图像信息的字典
    """
    try:
        # 获取二进制数据
        img_bytes = img_data['bytes']

        # 使用 PIL 和 BytesIO 打开图片
        image = Image.open(io.BytesIO(img_bytes))

        result = {
            "width": image.width,
            "height": image.height,
            "format": image.format
        }

        if embed:
            # 将图像编码为Base64
            buffered = io.BytesIO()
            image.save(buffered, format=image.format or "PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result["data"] = f"data:image/{image.format.lower() or 'png'};base64,{img_base64}"
        else:
            # 保存为文件
            filename = f"image_{index}.png"
            filepath = os.path.join(save_dir, filename)
            image.save(filepath)
            result["path"] = filepath

        return result

    except Exception as e:
        print(f"错误: 处理图像时出错: {str(e)}")
        return {"error": str(e)}

def parse_extra_info(extra_info):
    """
    解析extra_info列

    Args:
        extra_info: 可能是字典或字符串的extra_info值

    Returns:
        dict: 解析后的字典
    """
    try:
        # 如果extra_info已经是字典，直接返回
        if isinstance(extra_info, dict):
            return extra_info

        # 如果extra_info是字符串，尝试解析为JSON
        elif isinstance(extra_info, str):
            return json.loads(extra_info)

        # 其他情况返回空字典
        else:
            print(f"警告: 无法解析extra_info类型: {type(extra_info)}")
            return {}

    except Exception as e:
        print(f"错误: 解析extra_info时出错: {str(e)}")
        return {}

def custom_serializer(obj):
    """自定义JSON序列化器，处理NumPy数组和其他不可序列化的类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def clean_data(value):
    """清理数据，确保所有数据都可以被JSON序列化"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, dict):
        return {k: clean_data(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_data(v) for v in value]
    elif hasattr(value, '__dict__'):
        return clean_data(value.__dict__)
    else:
        return value

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 确保输出目录存在
    output_dir = ensure_dir(args.output_dir)

    # 如果不嵌入图像，确保图像目录存在
    if not args.embed_images:
        image_dir = ensure_dir(os.path.join(output_dir, args.image_dir))

    # 读取Parquet文件
    print(f"正在读取Parquet文件: {args.input_file}")
    try:
        df = pd.read_parquet(args.input_file, engine='pyarrow')

        # 如果指定了最大行数，只处理部分数据
        if args.max_rows and args.max_rows < len(df):
            df = df.head(args.max_rows)
            print(f"只处理前 {args.max_rows} 行数据")

    except Exception as e:
        print(f"错误: 无法读取Parquet文件: {str(e)}")
        return

    print(f"成功读取数据，共 {len(df)} 行")

    # 准备JSONL数据
    jsonl_data = []

    # 处理每一行数据
    print("正在处理数据并生成JSONL...")
    for idx, row in df.iterrows():
        try:
            # 基础数据
            record = {
                "id": idx,
                "problem": row.get('problem', ''),
                "answer": row.get('answer', ''),
                "data_source": row.get('data_source', ''),
                "prompt": row.get('prompt', ''),
                "ability": row.get('ability', ''),
                "reward_model": row.get('reward_model', '')
            }

            # 处理图像数据
            img_list = row.get('images', [])
            record["images"] = []

            for img_idx, img_data in enumerate(img_list):
                if args.embed_images:
                    img_result = process_image_data(img_data, f"{idx}_{img_idx}", embed=True)
                else:
                    img_result = process_image_data(img_data, f"{idx}_{img_idx}", image_dir, embed=False)

                record["images"].append(img_result)

            # 解析extra_info
            extra_info = row.get('extra_info', {})
            parsed_extra_info = parse_extra_info(extra_info)
            record["extra_info"] = parsed_extra_info

            # 清理记录，确保所有数据都可以被JSON序列化
            cleaned_record = {}
            for k, v in record.items():
                cleaned_record[k] = clean_data(v)

            # 添加到JSONL数据列表
            jsonl_data.append(cleaned_record)

        except Exception as e:
            print(f"错误: 处理第 {idx} 行时出错: {str(e)}")
            continue

    # 保存JSONL文件
    jsonl_path = os.path.join(output_dir, 'data.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in jsonl_data:
            try:
                f.write(json.dumps(record, ensure_ascii=False, default=custom_serializer) + '\n')
            except TypeError as e:
                print(f"错误: 无法序列化记录 {record.get('id', 'unknown')}: {str(e)}")
                # 尝试使用更严格的清理
                try:
                    # 创建一个新的清理后的记录
                    strict_cleaned_record = {}
                    for k, v in record.items():
                        # 只保留基本类型和字符串
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            strict_cleaned_record[k] = v
                        elif isinstance(v, (list, dict)):
                            # 尝试转换为字符串
                            strict_cleaned_record[k] = str(v)
                    f.write(json.dumps(strict_cleaned_record, ensure_ascii=False) + '\n')
                except Exception as e2:
                    print(f"严重错误: 无法序列化记录 {record.get('id', 'unknown')} 即使经过严格清理: {str(e2)}")

    print(f"处理完成!")
    print(f"JSONL文件已保存至: {jsonl_path}")

    if not args.embed_images:
        print(f"图像文件已保存至: {image_dir}")

    # 同时保存一份样本文件（前5条记录）
    sample_path = os.path.join(output_dir, 'sample.json')
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(jsonl_data[:5], f, ensure_ascii=False, indent=2, default=custom_serializer)

    print(f"样本数据已保存至: {sample_path}")

if __name__ == "__main__":
    main()
