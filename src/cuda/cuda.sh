#!/bin/bash

output_file="output.log"

while true; do
    # 运行 Python 脚本，并将输出重定向到文件
    python cuda.py >> "$output_file" &
    python_pid=$!
    
    # 等待 10 分钟
    sleep 600
    
    # 终止 Python 进程
    pkill -P $python_pid
    
    # 输出日志或者提示信息到文件，并添加时间戳
    echo "$(date): Python 脚本已运行 10 分钟，并已终止。将在下一个小时继续执行。" >> "$output_file"
    
    # 等待 50 分钟，总共等待 1 小时
    sleep 3000
done