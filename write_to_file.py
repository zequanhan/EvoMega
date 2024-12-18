import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio import motifs
from Bio.Seq import Seq
def generate_all_pwm_and_write_to_meme(tfbs_df, output_dir='/home/hanzequan/test_bectiral/operator_recongize/picture_10_phage_tfbs/EvoMega/new_score_system/motif_meme'):
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义最终输出文件路径
    meme_output_path = os.path.join(output_dir, 'final_motif.meme')
    
    # 初始化文件，写入头部信息
    with open(meme_output_path, 'w') as file:
        file.write('MEME version 4\n\n')
        file.write('ALPHABET= ACGT\n\n')
        file.write('strands: + -\n\n')
        file.write('Background letter frequencies\n')
        file.write('A 0.25 C 0.25 G 0.25 T 0.25\n\n')

    # 获取不同的编号
    unique_numbers = tfbs_df['Number'].unique()

    for number in unique_numbers:
        df_number = tfbs_df[tfbs_df['Number'] == number]
        unique_layers = df_number['Layer'].unique()

        # 遍历每个层，并生成motif
        for idx, layer in enumerate(unique_layers):
            df_layer = df_number[df_number['Layer'] == layer]
            sequences = [Seq(seq) for seq in df_layer['Sequence']]
            m = motifs.create(sequences)
            
            # 归一化PWM
            pwm = m.counts.normalize(pseudocounts={'A': 0, 'C': 0, 'G': 0, 'T': 0})
            
            # 将PWM矩阵转换为DataFrame
            pwm_df = pd.DataFrame({nucleotide: pwm[nucleotide] for nucleotide in 'ACGT'})
            
            # 确保pwm_df是有效的DataFrame
            if not isinstance(pwm_df, pd.DataFrame):
                raise TypeError(f"Expected pwm_df to be a pandas DataFrame, but got {type(pwm_df)}")

            # 将该motif写入最终文件，以追加模式
            write_motif_to_file_old(pwm_df, meme_output_path, layer, number, append=True)

def write_motif_to_file_old(pwm_df, file_path, motif_name, number, append=True):
    """
    将给定的PWM矩阵写入MEME格式的文件
    """
    # 确保pwm_df是有效的DataFrame
    if not isinstance(pwm_df, pd.DataFrame):
        raise TypeError(f"Expected pwm_df to be a pandas DataFrame, but got {type(pwm_df)}")

    # 获取PWM矩阵的宽度
    w = pwm_df.shape[0]
    mode = 'a' if append else 'w'
    
    # 打开文件（如果是第一次写入，mode是'w'，如果是追加则是'a'）
    with open(file_path, mode) as file:
        # 不再需要写入文件头部，因为在主函数中已经初始化过

        # 写入motif的基本信息
        file.write(f'MOTIF {number}_{motif_name}\n')
        file.write(f'letter-probability matrix: alength= 4 w= {w} nsites= 20 E= 0\n')

        # 将PWM矩阵转换为字符串并写入文件
        pwm_string = pwm_df.to_string(index=False, header=False)
        file.write(pwm_string + '\n\n')  # 添加换行符分隔不同的motif
def write_motif_to_file(pwm_df, file_path, motif_id, append=False):
    w = pwm_df.shape[0]
    mode = 'a' if append else 'w'
    with open(file_path, mode) as file:
        if not append:
            file.write('MEME version 4\n\n')
            file.write('ALPHABET= ACGT\n\n')
            file.write('strands: + -\n\n')
            file.write('Background letter frequencies\n')
            file.write('A 0.25 C 0.25 G 0.25 T 0.25\n\n')

        file.write(f'MOTIF {motif_id}\n')
        file.write(f'letter-probability matrix: alength= 4 w= {w} nsites= 20 E= 0\n')
        pwm_string = pwm_df.to_string(index=False, header=False)
        file.write(pwm_string + '\n\n')
def write_all_motifs_to_one_file(tfbs_df, combined_motifs_path):
    # 首先不在这里写入头部，让 write_motif_to_file 根据 append 来决定是否写头部
    # 你可以使用一个变量来判断是不是第一个motif
    first_write = True

    for num in tfbs_df['Number'].unique():
        df_num = tfbs_df[tfbs_df['Number'] == num]
        for layer in df_num['Layer'].unique():
            seqs = df_num[df_num['Layer'] == layer]['Sequence'].dropna().tolist()
            if not seqs:
                continue
            from Bio import motifs
            from Bio.Seq import Seq
            m = motifs.create([Seq(s) for s in seqs])
            pwm = m.counts.normalize(pseudocounts={'A':0,'C':0,'G':0,'T':0})
            pwm_df = pd.DataFrame({nuc: pwm[nuc] for nuc in "ACGT"})
            motif_id = f"{num}_{layer}"
            
            # 第一次写入时append=False，后面全部append=True
            write_motif_to_file(pwm_df, combined_motifs_path, motif_id, append=(not first_write))
            first_write = False
def extract_all_motif_ids(tfbs_df):
    return [f"{num}_{layer}" for num, layer in tfbs_df[['Number','Layer']].drop_duplicates().values]

def read_tomtom_results(file_path):

    # 添加了comment='#'
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith('Query_ID'):
            start_line = i
            break
   
    df = pd.read_csv(file_path, sep='\t', skiprows=start_line,comment='#')
    return df
def write_non_redundant_meme(tfbs_df, motif_ids, final_output):
    # 在写第一个 motif 时不使用 append，在后面追加时使用 append
    first_write = True

    # 遍历每个需要保留的 motif
    for motif_id in motif_ids:
        # motif_id 格式如 "Number_Layer"
        parts = motif_id.split('_', 1)
        if len(parts) < 2:
            # 无效的 motif_id，跳过
            continue
        num_str, layer = parts[0], parts[1]

        # 转换 num_str 为 int（假设 Number 是整数）
        try:
            num = int(num_str)
        except ValueError:
            # 无法解析 Number，跳过此 motif
            continue

        # 从 tfbs_df 中提取该 Number-Layer 对应的序列
        df_layer = tfbs_df[(tfbs_df['Number'] == num) & (tfbs_df['Layer'] == layer)]
        seqs = df_layer['Sequence'].dropna().tolist()
        if not seqs:
            # 没有有效序列，跳过
            continue

        # 构建 PWM
        from Bio import motifs
        from Bio.Seq import Seq
        m = motifs.create([Seq(s) for s in seqs])
        pwm = m.counts.normalize(pseudocounts={'A':0,'C':0,'G':0,'T':0})
        pwm_df = pd.DataFrame({nuc: pwm[nuc] for nuc in "ACGT"})

        # 使用 write_motif_to_file 函数写入
        # 根据前面函数的参数设定：first_number 对应 num，motif_name 对应 layer
        write_motif_to_file_old(pwm_df, final_output, layer, num, append=(not first_write))
        first_write = False
