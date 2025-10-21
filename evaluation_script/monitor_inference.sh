#!/bin/sh

MAX_RETRIES=3
TARGET_LINES=754
OUTPUT_FILE="./result_geoqa/results_model_name.jsonl"
MAIN_SCRIPT="sh run_infer.sh"
PROGRESS_LOG="progress.log"

retry_count=0
success=0

# 获取当前结果行数
get_current_lines() {
    if [ -f "$OUTPUT_FILE" ]; then
        wc -l < "$OUTPUT_FILE"
    else
        echo 0
    fi
}

# 记录进度
record_progress() {
    echo "尝试 $retry_count: 完成 $1 条数据" >> "$PROGRESS_LOG"
}

# 创建进度日志
echo "===== 开始监控任务 =====" > "$PROGRESS_LOG"
echo "目标行数: $TARGET_LINES" >> "$PROGRESS_LOG"
echo "最大重试次数: $MAX_RETRIES" >> "$PROGRESS_LOG"

# 初始行数
initial_lines=$(get_current_lines)
echo "初始行数: $initial_lines" >> "$PROGRESS_LOG"

while [ $retry_count -le $MAX_RETRIES ]; do
    current_lines=$(get_current_lines)
    echo "▶️ 启动推理脚本 (尝试: $(($retry_count+1))/$(($MAX_RETRIES+1))) | 当前进度: $current_lines/$TARGET_LINES"
    
    # 运行主脚本
    $MAIN_SCRIPT
    
    # 检查输出文件行数
    new_lines=$(get_current_lines)
    echo "��� 运行后行数: $new_lines/$TARGET_LINES"
    record_progress "$new_lines"
    
    # 检查是否完成
    if [ $new_lines -ge $TARGET_LINES ]; then
        echo "✅ 成功处理 $new_lines 条数据!"
        success=1
        break
    fi
    
    # 检查是否有进展
    if [ $new_lines -le $current_lines ]; then
        echo "⚠️ 警告: 本次运行未产生新数据"
    fi
    
    retry_count=$((retry_count+1))
    
    # 如果还有重试机会，等待后继续
    if [ $retry_count -le $MAX_RETRIES ]; then
        echo "⏳ 准备重试 (剩余重试次数: $(($MAX_RETRIES - $retry_count + 1)))"
        sleep 10
    fi
done

# 最终结果处理
final_lines=$(get_current_lines)
if [ $success -eq 1 ]; then
    echo "任务成功完成! 最终行数: $final_lines"
    echo "成功: 最终行数 $final_lines" >> "$PROGRESS_LOG"
    exit 0
else
    echo "❌ 已达最大重试次数 ($(($MAX_RETRIES+1)) 次), 最终行数: $final_lines"
    echo "失败: 最终行数 $final_lines" >> "$PROGRESS_LOG"
    exit 1
fi
