import subprocess
import os
from datetime import datetime
from tqdm import tqdm  # 导入 tqdm 库
import glob

# 获取所有 .gbk 文件的路径
input_files = glob.glob('/home/hanzequan/test_set_megadna/gbk_file/*.gbk')

# 输出目录
output_dir = '/home/hanzequan/test_set_megadna/output/'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 错误日志文件
log_file = os.path.join(output_dir, f"motif_analyzer_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# 打开日志文件
with open(log_file, 'w') as log:
    # 写入日志的开始时间
    log.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 使用 tqdm 包装 input_files 以显示进度条
    for input_file in tqdm(input_files, desc="Processing files", unit="file"):
        # 构造命令
        command = [
            'python3', 'motif_analyzer.py',
            '-i', input_file,
            '-o', output_dir
        ]

        try:
            # 清除 PERL5LIB 和 LD_LIBRARY_PATH 环境变量
            env = os.environ.copy()  # 复制当前环境变量
            env.pop('PERL5LIB', None)  # 删除 PERL5LIB
            env.pop('LD_LIBRARY_PATH', None)  # 删除 LD_LIBRARY_PATH

            # 打印当前正在运行的命令
            log.write(f"Running command: {' '.join(command)}\n")
            
            # 调用命令并捕获标准输出和错误
            result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
            
            # 如果命令成功，记录标准输出
            log.write(f"Standard Output:\n{result.stdout}\n")
            log.write(f"Standard Error:\n{result.stderr}\n")

        except subprocess.CalledProcessError as e:
            # 如果命令失败，记录错误
            log.write(f"Error occurred while running: {' '.join(command)}\n")
            log.write(f"Error Output:\n{e.stderr}\n")
            log.write(f"Error Code: {e.returncode}\n")
            log.write("\n" + "="*50 + "\n\n")

    # 记录结束时间
    log.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

