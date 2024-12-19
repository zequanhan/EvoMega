from Bio.Seq import Seq
import shutil

from Bio import SeqIO
import random
import sys
import re
import os
import pandas as pd
import numpy as np
from Bio import motifs
from Bio.Seq import Seq
from scipy import stats  # 确保导入scipy.stats
from sklearn.cluster import DBSCAN
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
sys.path.append('/home/public_new/web_and_evo_tree/genomics_site/genomics_app/mymodule/')
sys.path.append('/home/hanzequan/test_bectiral/rf_model/model1/algroasm')
from feature import com_seq_feature
from total_step_integrate_tfbs_and_promoter import GenomeAnalyzer
def complement_sequence(sequence):
    seq = Seq(sequence)
    return str(seq.complement())

# 定义生成反向序列的函数
def reverse_sequence(sequence):
    seq = Seq(sequence)
    return str(seq[::-1])

# 定义生成反向互补序列的函数
def reverse_complement_sequence(sequence):
    seq = Seq(sequence)
    return str(seq.reverse_complement())
def calculate_end(row):
    if row['stand'] == '+':
        return row['start'] + len(row['sequence']) - 1
    else:
        return row['start'] - len(row['sequence']) + 1

# 反转互补的函数
def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())

# 延伸序列的函数，包括反转互补处理

def extend_sequence(row, gbk_folder):
    gbk_path = f"{gbk_folder}/{row['accession']}.gbk"
    record = SeqIO.read(gbk_path, "genbank")
    
    if row['stand'] == '+':
        start_pos = max(0, row['start'] - 11)
        end_pos = min(len(record.seq), row['end'] + 10)
        extended_sequence = str(record.seq[start_pos:end_pos])
    else:
        start_pos = max(0, row['end'] - 10)
        end_pos = min(len(record.seq), row['start'] + 11)
        extended_sequence = str(record.seq[start_pos:end_pos])
        # 对负链数据进行反转互补处理
        extended_sequence = reverse_complement(extended_sequence)
    
    return extended_sequence
    
def extend_and_update(row, gbk_folder):
    gbk_path = f"{gbk_folder}/{row['accession']}.gbk"
    record = SeqIO.read(gbk_path, "genbank")
    
    if row['stand'] == '+':
        # 计算新的起始和终止位置
        new_start = max(0, row['start'] - 11)
        new_end = min(len(record.seq), row['end'] + 10)
        extended_seq = record.seq[new_start:new_end]
    else:
        # 对于负链，延伸并反转互补
        new_start = max(0, row['end'] - 10)
        new_end = min(len(record.seq), row['start'] + 11)
        extended_seq = record.seq[new_start:new_end].reverse_complement()
    
    # 更新行信息
    row['sequence'] = str(extended_seq)
    row['start'] = new_start + 1  # BioPython 序列位置从 0 开始，但通常表示时从 1 开始
    row['end'] = new_end
    return row
def combine_tomtom_results_with_score(results_df, result_path='/home/hanzequan/pwm_files/tomtom_results'):
    # 获取所有 tomtom_results_*.csv 文件
    csv_files = [os.path.join(result_path, f) for f in os.listdir(result_path) if f.startswith('tomtom_results_') and f.endswith('.csv')]
    
    # 读取所有 csv 文件并整合成一个 DataFrame
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 去除包含 NaN 值的行
    combined_df.dropna(inplace=True)
    
    # 创建 results_df 中的合并列用于匹配
    results_df['Combined_ID'] = results_df['Number'].astype(str) + '_' + results_df['Layer']
    
    # 创建 Score 列
    combined_df['Score'] = combined_df['Target_ID'].map(
        results_df.set_index('Combined_ID')['Average_Probability']
    )

    return combined_df


def random_interception_noncoding_extended(accession, seq_length, fasta_dir, gbk_dir):
    gbk_path = f"{gbk_dir}/{accession}.gbk"

    try:
        gbk_record = next(SeqIO.parse(gbk_path, "genbank"))
        genome_seq = str(gbk_record.seq)
        noncoding_regions = find_noncoding_regions(gbk_record, seq_length)
        
        if noncoding_regions:
            selected_region = random.choice(noncoding_regions)
            start_pos = random.randint(selected_region[0], selected_region[1] - seq_length)
            end_pos = start_pos + seq_length
            neg_seq = genome_seq[start_pos:end_pos]
            return neg_seq, start_pos, end_pos
        else:
            # 如果没有足够长的非编码区域，随机选择起始位置并截取，可能涉及编码区
            start_pos = random.randint(0, len(genome_seq) - seq_length)
            end_pos = start_pos + seq_length
            neg_seq = genome_seq[start_pos:end_pos]
            return neg_seq, start_pos, end_pos
    except Exception as e:
        print(f"Error processing {accession}: {e}")
    
    return None, None, None
def find_noncoding_regions(gbk_record, seq_length):
    """识别非编码区域，并返回一个包含(start, end)元组的列表"""
    noncoding_regions = []
    coding_regions = [feature for feature in gbk_record.features if feature.type == "CDS"]
    sorted_coding_regions = sorted(coding_regions, key=lambda x: x.location.start)

    last_end = 0
    for feature in sorted_coding_regions:
        start = int(feature.location.start)
        if last_end < start:  # 找到非编码区域
            noncoding_regions.append((last_end, start))
        last_end = int(feature.location.end)

    # 处理最后一个编码区域后的序列
    if last_end < len(gbk_record.seq):
        noncoding_regions.append((last_end, len(gbk_record.seq)))
    
    # 筛选出长度至少为seq_length的非编码区域
    suitable_regions = [region for region in noncoding_regions if region[1] - region[0] >= seq_length]
    return suitable_regions

def generate_wgri_negative_sample_with_start(seq_length, fasta_dir, accessions_list):
    """从整个基因组中随机截取相应长度的序列作为一个WGRI负样本，同时返回起始位置"""
    accession = random.choice(accessions_list)
    phage_fasta_path = f"{fasta_dir}/{accession}.fasta"
    try:
        seq_record = next(SeqIO.parse(phage_fasta_path, "fasta"))
        genome_seq = str(seq_record.seq)
        start_pos = random.randint(0, len(genome_seq) - seq_length)
        neg_seq = genome_seq[start_pos:start_pos + seq_length]
        return neg_seq, start_pos
    except:
        return None, None
def intra_group_disruption(seq, k=3):
    """随机交换子序列内的碱基位置来生成负样本"""
    def shuffle_subsequence(subseq):
        subseq_list = list(subseq)
        random.shuffle(subseq_list)
        return ''.join(subseq_list)
    
    subsequences = [shuffle_subsequence(seq[i:i+k]) for i in range(0, len(seq), k)]
    return ''.join(subsequences)
def inter_group_disruption(seq, k=3):
    """随机交换序列中的子序列位置来生成负样本"""
    subsequences = [seq[i:i+k] for i in range(0, len(seq), k)]
    random.shuffle(subsequences)
    return ''.join(subsequences)
def encode_sequences(sequences, max_len=50):
    # 核苷酸编码，加上一个用于填充的0编码
    nucleotide_to_int = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    encoded_seqs = []
    for seq in sequences:
        # 编码序列
        encoded_seq = [nucleotide_to_int.get(nuc, 0) for nuc in seq]
        # 如果序列长度小于max_len，则用0填充
        padded_seq = encoded_seq + [0] * (max_len - len(encoded_seq))
        encoded_seqs.append(padded_seq[:max_len])  # 确保序列长度不超过max_len
    return np.array(encoded_seqs)

def read_meme_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    motifs = []
    pwm_matrix = []
    current_motif = None

    for line in lines:
        if line.startswith('MOTIF'):
            if current_motif and pwm_matrix:
                motifs.append((current_motif, pwm_matrix))
                pwm_matrix = []
            current_motif = line.strip().split(' ')[1]
        elif line.startswith('letter-probability matrix'):
            continue
        elif current_motif and line.strip():
            pwm_matrix.append(list(map(float, line.strip().split())))

    if current_motif and pwm_matrix:
        motifs.append((current_motif, pwm_matrix))

    return motifs

def create_pwm_dataframe(motifs):
    data = {'motif_number': [], 'pwm': []}
    for motif, pwm in motifs:
        data['motif_number'].append(motif)
        data['pwm'].append(pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T']))
    
    return pd.DataFrame(data)

def combine_tomtom_results(result_path='/home/hanzequan/pwm_files/tomtom_results'):
    # 获取所有 tomtom_results_*.csv 文件
    csv_files = [os.path.join(result_path, f) for f in os.listdir(result_path) if f.startswith('tomtom_results_') and f.endswith('.csv')]
    
    # 读取所有 csv 文件并整合成一个 DataFrame
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 去除包含 NaN 值的行
    combined_df.dropna(inplace=True)
    
    return combined_df



def process_sequences(file_path, target_accession):
    with open(file_path, 'r') as file:
        content = file.read()

    cleaned_content = content.replace('..', ',')
    entries = cleaned_content.strip().split('>')[1:]

    header_pattern = re.compile(r'RefSeq:\s*([^,]+),.*?\((\d+)\s*[,.]\s*(\d+)\)')
    data = []

    for entry in entries:
        lines = entry.strip().split('\n')
        header = lines[0]
        sequence = ''.join(lines[1:]).strip()
        match = header_pattern.search(header)
        if match:
            accession = match.group(1).strip()
            start = match.group(2)
            end = match.group(3)
            data.append({
                'accession': accession,
                'start': start,
                'end': end,
                'sequence': sequence
            })

    df = pd.DataFrame(data)
    if target_accession not in df['accession'].values:
        print(f"Accession {target_accession} not found in the dataset.")
        return None
    
    sequences = df[df['accession'] == target_accession]['sequence'].tolist()
    min_length = min(len(seq) for seq in sequences)
    sequences_truncated = [seq[:min_length] for seq in sequences]
    sequences_truncated = [Seq(seq) for seq in sequences_truncated]
    m = motifs.create(sequences_truncated)
    pwm = m.counts.normalize(pseudocounts={'A': 0, 'C': 0, 'G': 0, 'T': 0})
    pwm_df_true = pd.DataFrame({nucleotide: pwm[nucleotide] for nucleotide in 'ACGT'})

    return pwm_df_true

def read_tomtom_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith('Query_ID'):
            start_line = i
            break
    
    df = pd.read_csv(file_path, sep='\t', skiprows=start_line,comment='#')
    return df
# 传统的tomtom
def run_tomtom_small_e_value(result_path, query_motif, target_motif):
    tomtom_command = [
        "/home/hanzequan/meme/bin/tomtom",
        "-thresh", "0.00001",
        "-oc", result_path,
        query_motif,
        target_motif,
        '-verbosity','1'
    ]
    
    try:
        subprocess.run(tomtom_command, check=True)
        # print("Tomtom analysis completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Tomtom: {e}")
        return None

    output_file = os.path.join(result_path, 'tomtom.tsv')
    return read_tomtom_results(output_file)



## 老代码 remove
def remove_redundant_sequences(tfbs_df, output_dir='/home/hanzequan/pwm_files', result_path='/home/hanzequan/pwm_files/tomtom_results'):
    if os.path.exists(result_path):
        # 清空文件夹
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    unique_numbers = tfbs_df['Number'].unique()
    query_motif_path = os.path.join(output_dir, f'motifs_{unique_numbers[1]}.meme')
    print('query_motif_path',query_motif_path)
    query_layers = set(tfbs_df[tfbs_df['Number'] == unique_numbers[1]]['Layer'])
    
    current_motif_path = os.path.join(result_path, 'current_motif.meme')
    os.system(f'cp {query_motif_path} {current_motif_path}')  # 将第一个文件复制为当前正在使用的motif文件
    print('current_motif_path',current_motif_path)
    for i in range(1, len(unique_numbers)):
        target_number = unique_numbers[i]
        target_motif_path = os.path.join(output_dir, f'motifs_{target_number}.meme')
        
        tomtom_results = run_tomtom_small_e_value(result_path, current_motif_path, target_motif_path)
        
        if tomtom_results is not None:
            # 保存每个 tomtom_results 为 CSV 文件
            csv_filename = os.path.join(result_path, f'tomtom_results_{target_number}.csv')
            tomtom_results.to_csv(csv_filename, index=False)

            all_target_layers = set()
            for _, row in tomtom_results.iterrows():
                if pd.notna(row['Target_ID']):
                    target_layer_parts = row['Target_ID'].split('_')
                    target_layer = '_'.join(target_layer_parts[-2:])  # 拼接后两个部分
                    all_target_layers.add(target_layer)
            
            new_layers = query_layers - all_target_layers
            for target_layer in new_layers:
                df_target_layer = tfbs_df[(tfbs_df['Number'] == target_number) & (tfbs_df['Layer'] == target_layer)]
                sequences = [Seq(seq) for seq in df_target_layer['Sequence']]
                
                if sequences:
                    m = motifs.create(sequences)
                    pwm = m.counts.normalize(pseudocounts={'A': 0, 'C': 0, 'G': 0, 'T': 0})
                    pwm_df = pd.DataFrame({nucleotide: pwm[nucleotide] for nucleotide in 'ACGT'})
                    write_motif_to_file(pwm_df, current_motif_path, target_layer, target_number, append=True)
                    query_layers.add(target_layer)
# def remove_redundant_sequences(tfbs_df, output_dir='/home/hanzequan/pwm_files', result_path='/home/hanzequan/pwm_files/tomtom_results'):
#     import os
#     import shutil

#     if os.path.exists(result_path):
#         # 清空文件夹
#         shutil.rmtree(result_path)
#     os.makedirs(result_path)

#     unique_numbers = tfbs_df['Number'].unique()
#     print('unique_numbers', unique_numbers)
    
#     # 创建包含伪数据的初始 motif 文件
#     current_motif_path = os.path.join(result_path, 'current_motif.meme')
#     with open(current_motif_path, 'w') as f:
#         f.write("MEME version 4\n\n")
#         f.write("ALPHABET= ACGT\n\n")
#         f.write("strands: + -\n\n")

#         # 写入 fake_motif_1
#         f.write("MOTIF fake_motif_1\n")
#         f.write("letter-probability matrix: alength= 4 w= 16 nsites= 20 E= 0\n")
#         for _ in range(16):
#             f.write("0.25 0.25 0.25 0.25\n")
#         f.write("\n")

#         # 写入 fake_motif_2
#         f.write("MOTIF fake_motif_2\n")
#         f.write("letter-probability matrix: alength= 4 w= 16 nsites= 20 E= 0\n")
#         for _ in range(16):
#             f.write("0.25 0.25 0.25 0.25\n")
#         f.write("\n")

#         # 写入 fake_motif_3
#         f.write("MOTIF fake_motif_3\n")
#         f.write("letter-probability matrix: alength= 4 w= 16 nsites= 20 E= 0\n")
#         for _ in range(16):
#             f.write("0.25 0.25 0.25 0.25\n")
#         f.write("\n")

#     # 初始化 query_layers 集合
#     query_layers = set()

#     print('current_motif_path', current_motif_path)
#     for i in range(len(unique_numbers)):
#         target_number = unique_numbers[i]
#         target_motif_path = os.path.join(output_dir, f'motifs_{target_number}.meme')
        
#         tomtom_results = run_tomtom_small_e_value(result_path, current_motif_path, target_motif_path)
        
#         if tomtom_results is not None:
#             # 保存每个 tomtom_results 为 CSV 文件
#             csv_filename = os.path.join(result_path, f'tomtom_results_{target_number}.csv')
#             tomtom_results.to_csv(csv_filename, index=False)

#             all_target_layers = set()
#             for _, row in tomtom_results.iterrows():
#                 if pd.notna(row['Target_ID']):
#                     target_layer_parts = row['Target_ID'].split('_')
#                     target_layer = '_'.join(target_layer_parts[-2:])  # 拼接后两个部分
#                     all_target_layers.add(target_layer)
            
#             # 获取目标 Number 的所有 Layer
#             target_layers = set(tfbs_df[tfbs_df['Number'] == target_number]['Layer'])
#             new_layers = target_layers - all_target_layers

#             for target_layer in new_layers:
#                 df_target_layer = tfbs_df[(tfbs_df['Number'] == target_number) & (tfbs_df['Layer'] == target_layer)]
#                 sequences = [Seq(seq) for seq in df_target_layer['Sequence']]
                
#                 if sequences:
#                     m = motifs.create(sequences)
#                     pwm = m.counts.normalize(pseudocounts={'A': 0.1, 'C': 0.1, 'G': 0.1, 'T': 0.1})
#                     pwm_df = pd.DataFrame({nucleotide: pwm[nucleotide] for nucleotide in 'ACGT'})
#                     write_motif_to_file(pwm_df, current_motif_path, target_layer, target_number, append=True)
#                     query_layers.add(target_layer)
#         else:
#             print(f"No Tomtom results for target_number {target_number}")


def generate_all_pwm_and_write_to_meme(tfbs_df, output_dir='/home/hanzequan/pwm_files/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    unique_numbers = tfbs_df['Number'].unique()
    for number in unique_numbers:
        df_number = tfbs_df[tfbs_df['Number'] == number]
        unique_layers = df_number['Layer'].unique()
        meme_output_path = os.path.join(output_dir, f'motifs_{number}.meme')

        for idx, layer in enumerate(unique_layers):
            df_layer = df_number[df_number['Layer'] == layer]
            sequences = [Seq(seq) for seq in df_layer['Sequence']]
            m = motifs.create(sequences)
            pwm = m.counts.normalize(pseudocounts={'A': 0, 'C': 0, 'G': 0, 'T': 0})
            pwm_df = pd.DataFrame({nucleotide: pwm[nucleotide] for nucleotide in 'ACGT'})
            write_motif_to_file(pwm_df, meme_output_path, layer, number, append=(idx != 0))

def write_motif_to_file(pwm_df, file_path, motif_name, first_number, append=False):
    w = pwm_df.shape[0]
    mode = 'a' if append else 'w'
    with open(file_path, mode) as file:
        if not append:
            file.write('MEME version 4\n\n')
            file.write('ALPHABET= ACGT\n\n')
            file.write('strands: + -\n\n')
            file.write('Background letter frequencies\n')
            file.write('A 0.25 C 0.25 G 0.25 T 0.25\n\n')
        
        file.write(f'MOTIF {first_number}_{motif_name}\n')
        file.write(f'letter-probability matrix: alength= 4 w= {w} nsites= 20 E= 0\n')
        pwm_string = pwm_df.to_string(index=False, header=False)
        file.write(pwm_string + '\n\n')  # 添加换行符分隔不同的motif
def build_motif_matrices(directory, sequence_count_occurrences=None):
    if sequence_count_occurrences is None:
        sequence_count_occurrences = {}  # 确保 sequence_count_occurrences 在每次运行时初始化为空字典

    all_motifs_data = pd.DataFrame()
    found_xml = False  # 标记是否找到至少一个meme.xml文件

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'meme.xml':
                found_xml = True
                xml_file = os.path.join(root, file)
                txt_file = os.path.join(os.path.dirname(root), os.path.basename(root) + '.txt')

                layer_dir_name = os.path.basename(root)
                fasta_file = os.path.join(os.path.dirname(root), f"{layer_dir_name}_concatenated.fasta")
                try:
                    with open(txt_file, 'r') as txt:
                        for line in txt.readlines():
                            line = line.strip()
                            if line:
                                sequence_ids = line.split(',')
                                sequence_count = len(sequence_ids)
                                if sequence_count in sequence_count_occurrences:
                                    sequence_count_occurrences[sequence_count] += 1
                                else:
                                    sequence_count_occurrences[sequence_count] = 1  # 从1开始计数
                except FileNotFoundError:
                    print(f"Warning: Corresponding text file not found: {txt_file}")

                with open(xml_file) as f:
                    meme_record = motifs.parse(f, "meme")

                # Read the fasta file
                sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

                motif_index = 1
                for motif in meme_record:
                    motifs_data = []
                    for instance in motif.instances:
                        sequence_name = instance.sequence_name
                        id = motif.id
                        consensus = motif.name
                        e_value = motif.evalue
                        num_occurrences = len(motif.instances)
                        suffix = f".{sequence_count_occurrences[sequence_count]}" if sequence_count_occurrences[sequence_count] > 1 else ""

                        start = instance.start  # 从1开始计数
                        end = start + len(str(instance)) - 1

                        # Extend Start and End by 10bp
                        start_extend = max(1, start - 10)  # 确保从1开始
                        end_extend = end + 10

                        # Extract the extended sequence
                        full_sequence = sequences[sequence_name].seq
                        extend_sequence = full_sequence[start_extend-1:end_extend]  # 调整为从1开始

                        if instance.strand == '-':
                            start, end = end, start
                            start_extend, end_extend = end_extend, start_extend
                            extend_sequence = extend_sequence.reverse_complement()

                        motif_data = {
                            "Number": f"{sequence_count}{suffix}",
                            "Layer": id,
                            "Strand": instance.strand,
                            "Start": start,
                            "End": end,
                            "Start_extend": start_extend,
                            "End_extend": end_extend,
                            "p-value": instance.pvalue,
                            "e-value": e_value,
                            "Sequence": str(instance),
                            "Motif": consensus,
                            "Extend_sequence": str(extend_sequence)
                        }
                        motifs_data.append(motif_data)
                    if motifs_data:
                        motifs_df = pd.DataFrame(motifs_data)
                        all_motifs_data = pd.concat([all_motifs_data, motifs_df], ignore_index=True)
                    motif_index += 1

    if not found_xml:
        raise FileNotFoundError(f"No MEME output file found in directory {directory} or its subdirectories.")

    # 重置变量
    sequence_count_occurrences.clear()

    return all_motifs_data

def rank_query_ids_by_average_score(combined_df):
    # 计算每个 Query_ID 的序列数量
    query_counts = combined_df['Query_ID'].value_counts()
    
    # 确定序列数量的阈值
    threshold = combined_df.shape[0] / 50
    
    # 过滤掉序列数量少于阈值的 Query_ID
    valid_query_ids = query_counts[query_counts >= threshold].index
    
    # 计算每个有效 Query_ID 的平均 Score
    valid_combined_df = combined_df[combined_df['Query_ID'].isin(valid_query_ids)]
    average_scores = valid_combined_df.groupby('Query_ID')['Score'].mean()
    
    # 对平均分进行排名
    ranked_scores = average_scores.rank(ascending=False, method='min').astype(int)
    
    # 创建一个 DataFrame 来保存排名结果
    ranked_df = pd.DataFrame({
        'Query_ID': average_scores.index,
        'Average_Score': average_scores.values,
        'Rank': ranked_scores.values
    }).sort_values(by='Rank')
    
    # 为每个 Query_ID 添加对应的 Target_ID
    ranked_df = ranked_df.merge(combined_df[['Query_ID', 'Target_ID']], on='Query_ID', how='left')
    
    return ranked_df


# 更加科学的寻找到排名的函数
# def rank_query_ids_by_average_score(combined_df):
#     query_counts = combined_df['Query_ID'].value_counts()

#     # 设置一个合理的阈值来过滤掉稀少的 Query_ID
#     threshold = max(combined_df.shape[0] / 50, query_counts.quantile(0.05))
    
#     # 过滤掉序列数量少于阈值的 Query_ID
#     valid_query_ids = query_counts[query_counts >= threshold].index
    
#     # 计算每个有效 Query_ID 的平均分数、标准差
#     valid_combined_df = combined_df[combined_df['Query_ID'].isin(valid_query_ids)]
#     score_stats = valid_combined_df.groupby('Query_ID')['Score'].agg(['mean', 'std', 'count'])
    
#     # 引入权重因子来调整评分，例如用平均分数、标准差和样本量的组合来排名
#     score_stats['weighted_score'] = (score_stats['mean'] * 0.7) + ((1 / score_stats['std']) * 0.2) + (score_stats['count'] * 0.1)
    
#     # 根据加权分数进行排名
#     score_stats['Rank'] = score_stats['weighted_score'].rank(ascending=False, method='min').astype(int)
    
#     # 创建一个 DataFrame 来保存排名结果
#     ranked_df = score_stats[['mean', 'weighted_score', 'Rank']].reset_index().rename(columns={'mean': 'Average_Score'})
    
#     # 为每个 Query_ID 添加对应的 Target_ID，去重以确保唯一性
#     ranked_df = ranked_df.merge(combined_df[['Query_ID', 'Target_ID']].drop_duplicates(), on='Query_ID', how='left')
    
    return ranked_df
def adjust_top_query_scores(combined_df, ranked_df):
    # 根据 Rank 排序并去重，获取排名前3的唯一 Query_ID
    top_3_query_ids = ranked_df.drop_duplicates(subset=['Query_ID']).nsmallest(3, 'Rank')['Query_ID'].unique()
    print('adjust_top_query_scores',adjust_top_query_scores)
    # multipliers = [1.5, 1.4, 1.3] #正常打分
    multipliers = [1, 1, 1] #阴性对照 不打分
    
    # 调整排名前3的 Query_ID 的 Score
    for query_id, multiplier in zip(top_3_query_ids, multipliers):
        combined_df.loc[combined_df['Query_ID'] == query_id, 'Score'] *= multiplier
    
    # 将 Rank 列添加到 combined_df 中
    ranked_unique_df = ranked_df.drop_duplicates(subset=['Query_ID'])
    combined_df = combined_df.merge(ranked_unique_df[['Query_ID', 'Rank']], on='Query_ID', how='left')
    
    return combined_df

def construct_target_query_score_matrix(adjusted_combined_df, results_df):
    # 构建一个仅包含 'Target_ID', 'Score' 和 'Rank' 列的矩阵
    target_score_matrix = adjusted_combined_df[['Target_ID', 'Score', 'Rank']]
    # 如果 'Target_ID' 是重复的，只保留最高分数的记录
    target_score_matrix = target_score_matrix.loc[target_score_matrix.groupby('Target_ID')['Score'].idxmax()].reset_index(drop=True)
    
    # 构建一个仅包含 'Query_ID', 'Score' 和 'Rank' 列的矩阵
    query_score_matrix = adjusted_combined_df[['Query_ID', 'Score', 'Rank']]
    # 如果 'Query_ID' 是重复的，只保留最高分数的记录
    query_score_matrix = query_score_matrix.loc[query_score_matrix.groupby('Query_ID')['Score'].idxmax()].reset_index(drop=True)
    
    # 将 'Query_ID' 矩阵添加到 'Target_ID' 矩阵中
    query_score_matrix = query_score_matrix.rename(columns={'Query_ID': 'Target_ID'})
    combined_score_matrix = pd.concat([target_score_matrix, query_score_matrix], ignore_index=True)
    
    # 处理重复的 'Target_ID'，只保留最高分数的记录
    combined_score_matrix = combined_score_matrix.loc[combined_score_matrix.groupby('Target_ID')['Score'].idxmax()].reset_index(drop=True)
    
    # 检查在 results_df['Combined_ID'] 中存在但不在 combined_score_matrix['Target_ID'] 中的内容
    missing_ids = results_df['Combined_ID'][~results_df['Combined_ID'].isin(combined_score_matrix['Target_ID'])]
    
    # 添加这些内容到 combined_score_matrix
    new_rows = []
    for missing_id in missing_ids:
        new_row = {
            'Target_ID': missing_id,
            'Score': results_df.loc[results_df['Combined_ID'] == missing_id, 'Average_Probability'].values[0],
            'Rank': None  # 因为这些ID不在 adjusted_combined_df 中，所以 Rank 为 None
        }
        new_rows.append(new_row)
    
    if new_rows:
        combined_score_matrix = pd.concat([combined_score_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    
    return combined_score_matrix
def run_tomtom_high_e_value(result_path, query_motif, target_motif):
    # 构建命令
    tomtom_command = [
        "/home/hanzequan/meme/bin/tomtom",
        "-thresh", "50",  # 使用阈值参数，设定阈值为0.1
        "-evalue",        # 表示使用E-value作为阈值的基准
        # "-min-overlap",'100',
        "-oc", result_path,
        query_motif,
        target_motif,
        '-verbosity','1'

    ]
    
    # 运行 tomtom 命令
    try:
        subprocess.run(tomtom_command, check=True)
        # print("Tomtom analysis completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Tomtom: {e}")
        return None

    # 解析输出结果
    output_file = os.path.join(result_path, 'tomtom.tsv')
    return read_tomtom_results(output_file)
def write_motif_to_file_for_more_sequence_compare(pwm_df, file_path, motif_name, append=False):
    # 确保目录存在
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 如果文件不存在，强制覆盖模式，写入新的文件头
    if not os.path.exists(file_path):
        append = False  # 强制覆盖，写入文件头

    w = pwm_df.shape[0]  # PWM 矩阵的宽度
    mode = 'a' if append else 'w'  # 如果需要追加写入，使用 'a' 模式，否则使用 'w' 模式覆盖
    
    try:
        with open(file_path, mode) as file:
            # 如果是覆盖模式（文件首次创建），写入文件头
            if not append:
                file.write('MEME version 4\n\n')
                file.write('ALPHABET= ACGT\n\n')
                file.write('strands: + -\n\n')
                file.write('Background letter frequencies:\n')
                file.write('A 0.25 C 0.25 G 0.25 T 0.25\n\n')
            
            # 写入motif
            file.write(f'MOTIF {motif_name}\n')
            file.write(f'letter-probability matrix: alength= 4 w= {w} nsites= 20 E= 0\n')
            
            # 写入 PWM 矩阵数据
            pwm_string = pwm_df.to_string(index=False, header=False)
            file.write(pwm_string + '\n\n')  # 每个motif之间添加空行以分隔

        print(f"Successfully wrote motif {motif_name} to {file_path}")

    except Exception as e:
        print(f"Error writing motif {motif_name} to {file_path}: {e}")
        return False  # 返回 False 表示写入失败

    return True  # 返回 True 表示写入成功


def generate_all_pwm_and_write_to_meme_bidui(tfbs_df, output_file='/home/hanzequan/pwm_files/all_motifs.meme'):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    unique_numbers = tfbs_df['Number'].unique()
    first_write = True  # 标记是否为第一次写入文件
    
    for number in unique_numbers:
        df_number = tfbs_df[tfbs_df['Number'] == number]
        unique_layers = df_number['Layer'].unique()
        
        for idx, layer in enumerate(unique_layers):
            df_layer = df_number[df_number['Layer'] == layer]
            sequences = [Seq(seq) for seq in df_layer['Sequence']]
            if sequences:  # 确保序列不为空
                m = motifs.create(sequences)
                pwm = m.counts.normalize(pseudocounts={'A': 0, 'C': 0, 'G': 0, 'T': 0})
                pwm_df = pd.DataFrame({nucleotide: pwm[nucleotide] for nucleotide in 'ACGT'})
                
                # 写入motif文件
                success = write_motif_to_file_for_more_sequence_compare(pwm_df, output_file, f'{number}_{layer}', append=not first_write)
                if not success:
                    print(f"Skipping motif {number}_{layer} due to write error.")
                    continue  # 跳过错误的写入

                first_write = False  # 之后的写入都设置为追加模式

def process_result_df(result_df, min_p_value=1e-12):
    result_df = result_df.dropna(subset=['Target_ID'])  # 删除 Target_ID 列中为 NaN 的行
    result_df['Int_Part_1'] = result_df['Target_ID'].apply(lambda x: int(x.split('_')[0]) if isinstance(x, str) else None)
    result_df = result_df.dropna(subset=['Int_Part_1'])  # 删除 Int_Part_1 列中为 NaN 的行
    result_df = result_df.loc[result_df.groupby('Int_Part_1')['p-value'].idxmin()]
    
    # 将 p-value 转换为 z 值，p-value 小于 min_p_value 的设置为 min_p_value
    result_df['z-value'] = result_df['p-value'].apply(lambda p: stats.norm.ppf(1 - max(p, min_p_value)))
    
    return result_df

# def plot_scores(combined_df, processed_result_df, distance_result_df,distance_weight=2):
#     # 提取 Target_ID 中的整数部分并创建新的列用于分组
#     combined_df['Int_Part_1'] = combined_df['Target_ID'].apply(lambda x: int(x.split('_')[0]) if isinstance(x, str) else None)
#     combined_df = combined_df.dropna(subset=['Int_Part_1'])  # 删除 Int_Part_1 列中为 NaN 的行
    
#     # 将 distance_result_df 中的 Ratio 添加到 combined_df 的 Score 上
#     for idx, row in combined_df.iterrows():
#         target_id = row['Target_ID']
#         if target_id in distance_result_df['Number_Layer'].values:
#             ratio = distance_result_df.loc[distance_result_df['Number_Layer'] == target_id, 'Ratio'].values[0]
#             ### 修改位置距离权重
#             combined_df.at[idx, 'Score'] += ratio*distance_weight
    
#     # 应用权重调整排名前3的 Query_ID 的 Score
#     for rank, multiplier in zip([1, 2, 3,4], [1.6, 1.5, 1.4,1.3]):
#         target_ids = combined_df[combined_df['Rank'] == rank]['Target_ID'].unique()
#         combined_df.loc[combined_df['Target_ID'].isin(target_ids), 'Score'] *= multiplier
    
#     # 计算每个 Int_Part_1 的最高分
#     max_scores = combined_df.groupby('Int_Part_1')['Score'].max().reset_index()
    
#     # 创建相对刻度
#     max_scores['Relative_Index'] = range(len(max_scores))
    
#     # 创建折线图
#     fig, ax = plt.subplots(figsize=(22, 6))
#     ax.plot(max_scores['Relative_Index'], max_scores['Score'], marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=6)
#     ax.set_xlabel('Number')
#     ax.set_ylabel('Highest Score')
#     ax.set_title('Highest Scores ' + accession)
#     ax.set_xticks(ticks=max_scores['Relative_Index'])
#     ax.set_xticklabels(max_scores['Int_Part_1'], rotation=0, fontsize=6)  # 字体大小调整为8
#     ax.grid(True)
    
#     # # 添加红点，z-value 越大，颜色越深
#     norm = plt.Normalize(processed_result_df['z-value'].min(), processed_result_df['z-value'].max())
#     cmap = plt.get_cmap('coolwarm')  # 使用从红到蓝的颜色渐变
    
#     for idx, row in processed_result_df.iterrows():
#         if pd.notna(row['z-value']):
#             int_part_1 = row['Int_Part_1']
#             if int_part_1 in max_scores['Int_Part_1'].values:
#                 relative_index = max_scores.loc[max_scores['Int_Part_1'] == int_part_1, 'Relative_Index'].values[0]
#                 highest_score = max_scores.loc[max_scores['Int_Part_1'] == int_part_1, 'Score'].values[0]
#                 ax.scatter(relative_index, highest_score, color=cmap(norm(row['z-value'])), s=50, zorder=5)
    
#     # 添加颜色梯度标签
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label('z-value')
    
#     plt.show()
## 批量构图【0，1，2】
def plot_scores(combined_df, processed_result_df, distance_result_df, distance_weight=2):
    # 提取 Target_ID 中的整数部分并创建新的列用于分组
    combined_df['Int_Part_1'] = combined_df['Target_ID'].apply(lambda x: int(x.split('_')[0]) if isinstance(x, str) else None)
    combined_df = combined_df.dropna(subset=['Int_Part_1'])  # 删除 Int_Part_1 列中为 NaN 的行
    
    # 将 distance_result_df 中的 Ratio 添加到 combined_df 的 Score 上
    for idx, row in combined_df.iterrows():
        target_id = row['Target_ID']
        if target_id in distance_result_df['Number_Layer'].values:
            ratio = distance_result_df.loc[distance_result_df['Number_Layer'] == target_id, 'Ratio'].values[0]
            # 修改位置距离权重
            combined_df.at[idx, 'Score'] += ratio * distance_weight
    
    # 应用权重调整排名前4的 Query_ID 的 Score
    for rank, multiplier in zip([1, 2, 3, 4], [1.6, 1.5, 1.4, 1.3]):
        target_ids = combined_df[combined_df['Rank'] == rank]['Target_ID'].unique()
        combined_df.loc[combined_df['Target_ID'].isin(target_ids), 'Score'] *= multiplier
    
    # 计算每个 Int_Part_1 的最高分
    max_scores = combined_df.groupby('Int_Part_1')['Score'].max().reset_index()
    
    # 创建相对刻度
    max_scores['Relative_Index'] = range(len(max_scores))
    
    # 创建折线图
    fig, ax = plt.subplots(figsize=(22, 6))
    ax.plot(max_scores['Relative_Index'], max_scores['Score'], marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=6)
    ax.set_xlabel('Number')
    ax.set_ylabel('Highest Score')
    ax.set_title(f'Highest Scores {accession} with distance_weight={distance_weight}')
    ax.set_xticks(ticks=max_scores['Relative_Index'])
    ax.set_xticklabels(max_scores['Int_Part_1'], rotation=0, fontsize=6)  # 字体大小调整为8
    ax.grid(True)
    
    # 添加红点，z-value 越大，颜色越深
    norm = plt.Normalize(processed_result_df['z-value'].min(), processed_result_df['z-value'].max())
    cmap = plt.get_cmap('coolwarm')  # 使用从红到蓝的颜色渐变
    
    for idx, row in processed_result_df.iterrows():
        if pd.notna(row['z-value']):
            int_part_1 = row['Int_Part_1']
            if int_part_1 in max_scores['Int_Part_1'].values:
                relative_index = max_scores.loc[max_scores['Int_Part_1'] == int_part_1, 'Relative_Index'].values[0]
                highest_score = max_scores.loc[max_scores['Int_Part_1'] == int_part_1, 'Score'].values[0]
                ax.scatter(relative_index, highest_score, color=cmap(norm(row['z-value'])), s=50, zorder=5)
    
    # 添加颜色梯度标签
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('z-value')
    
    plt.show()

## 基础版计算距离函数
def calculate_ratio(extend_meme_10):
    results = []

    # 获取所有的唯一 Number 和 Layer
    numbers = extend_meme_10['Number'].unique()
    layers = extend_meme_10['Layer'].unique()

    for number in numbers:
        for layer in layers:
            # 筛选出特定 Number 和 Layer 的数据
            filtered_data = extend_meme_10[(extend_meme_10['Number'] == number) & (extend_meme_10['Layer'] == layer)]
            
            if filtered_data.empty:
                continue
            
            # 准备数据点，根据 Strand 来决定位置
            positions = []
            for index, row in filtered_data.iterrows():
                if row['Strand'] == '+':
                    positions.append(row['End'])
                else:
                    positions.append(row['Start'])

            positions = np.array(positions).reshape(-1, 1)

            # 使用 DBSCAN 进行聚类
            dbscan = DBSCAN(eps=40, min_samples=2).fit(positions)
            labels = dbscan.labels_

            # 将聚类结果添加回原数据
            filtered_data['Cluster'] = labels

            # 找到接近程度在 50bp 以内的序列对
            close_sequences = []
            for cluster in set(labels):
                if cluster != -1:
                    cluster_data = filtered_data[filtered_data['Cluster'] == cluster]
                    for i, row1 in cluster_data.iterrows():
                        for j, row2 in cluster_data.iterrows():
                            if i < j:
                                close_sequences.append({
                                    'Sequence_1': row1['Sequence'],
                                    'Start_1': row1['Start'],
                                    'End_1': row1['End'],
                                    'Strand_1': row1['Strand'],
                                    'Sequence_2': row2['Sequence'],
                                    'Start_2': row2['Start'],
                                    'End_2': row2['End'],
                                    'Strand_2': row2['Strand'],
                                    'Distance': abs((row1['End'] if row1['Strand'] == '+' else row1['Start']) - (row2['Start'] if row2['Strand'] == '+' else row2['End']))
                                })

            # 计算 df_close_sequences.shape[0]/filtered_data.shape[0]
            ratio = len(close_sequences) / len(filtered_data)
            
            # 存储结果
            results.append({
                'Number': int(number),
                'Number_Layer': f'{number}_{layer}',
                'Ratio': ratio
            })

    # 将结果转换为 DataFrame
    distance_result_df = pd.DataFrame(results)
    return distance_result_df
## 加权版距离函数，距离越近，分数越高
def calculate_ratio_distance(extend_meme_10):
    results = []

    # 获取所有的唯一 Number 和 Layer
    numbers = extend_meme_10['Number'].unique()
    layers = extend_meme_10['Layer'].unique()

    for number in numbers:
        for layer in layers:
            # 筛选出特定 Number 和 Layer 的数据
            filtered_data = extend_meme_10[(extend_meme_10['Number'] == number) & (extend_meme_10['Layer'] == layer)]
            
            if filtered_data.empty:
                continue
            
            # 准备数据点，根据 Strand 来决定位置
            positions = []
            for index, row in filtered_data.iterrows():
                if row['Strand'] == '+':
                    positions.append(row['End'])
                else:
                    positions.append(row['Start'])

            positions = np.array(positions).reshape(-1, 1)

            # 使用 DBSCAN 进行聚类
            dbscan = DBSCAN(eps=40, min_samples=2).fit(positions)
            labels = dbscan.labels_

            # 将聚类结果添加回原数据
            filtered_data['Cluster'] = labels

            # 找到接近程度在 50bp 以内的序列对，并赋予不同的权重
            weighted_scores = []
            for cluster in set(labels):
                if cluster != -1:
                    cluster_data = filtered_data[filtered_data['Cluster'] == cluster]
                    for i, row1 in cluster_data.iterrows():
                        for j, row2 in cluster_data.iterrows():
                            if i < j:
                                distance = abs((row1['End'] if row1['Strand'] == '+' else row1['Start']) - 
                                               (row2['Start'] if row2['Strand'] == '+' else row2['End']))
                                # 根据距离范围赋予权重
                                if distance <= 20:
                                    weight = 2  # 20bp 内给予更高的权重
                                elif distance <= 40:
                                    weight = 1  # 20-40bp 范围内给予标准权重
                                else:
                                    weight = 0  # 超过40bp 不计入

                                weighted_scores.append(weight)

            # 计算加权比率
            ratio = sum(weighted_scores) / len(filtered_data) if len(filtered_data) > 0 else 0
            
            # 存储结果
            results.append({
                'Number': int(number),
                'Number_Layer': f'{number}_{layer}',
                'Ratio': ratio
            })

    # 将结果转换为 DataFrame
    distance_result_df = pd.DataFrame(results)
    return distance_result_df

## 基于聚类数量增加权重
def calculate_weighted_ratio(extend_meme_10):
    results = []

    # 获取所有的唯一 Number 和 Layer
    numbers = extend_meme_10['Number'].unique()
    layers = extend_meme_10['Layer'].unique()

    for number in numbers:
        for layer in layers:
            # 筛选出特定 Number 和 Layer 的数据
            filtered_data = extend_meme_10[(extend_meme_10['Number'] == number) & (extend_meme_10['Layer'] == layer)]
            
            if filtered_data.empty:
                continue
            
            # 准备数据点，根据 Strand 来决定位置
            positions = []
            for index, row in filtered_data.iterrows():
                if row['Strand'] == '+':
                    positions.append(row['End'])
                else:
                    positions.append(row['Start'])

            positions = np.array(positions).reshape(-1, 1)

            # 使用 DBSCAN 进行聚类
            dbscan = DBSCAN(eps=40, min_samples=2).fit(positions)
            labels = dbscan.labels_

            # 将聚类结果添加回原数据
            filtered_data['Cluster'] = labels

            # 找到接近程度在 50bp 以内的序列对，并根据聚类大小进行加权计算
            close_sequences = []
            cluster_sizes = {label: (labels == label).sum() for label in set(labels) if label != -1}
            for cluster, size in cluster_sizes.items():
                if cluster != -1:
                    cluster_data = filtered_data[filtered_data['Cluster'] == cluster]
                    for i, row1 in cluster_data.iterrows():
                        for j, row2 in cluster_data.iterrows():
                            if i < j:
                                distance = abs((row1['End'] if row1['Strand'] == '+' else row1['Start']) - 
                                               (row2['Start'] if row2['Strand'] == '+' else row2['End']))
                                if distance <= 50:
                                    close_sequences.append({
                                        'Sequence_1': row1['Sequence'],
                                        'Start_1': row1['Start'],
                                        'End_1': row1['End'],
                                        'Strand_1': row1['Strand'],
                                        'Sequence_2': row2['Sequence'],
                                        'Start_2': row2['Start'],
                                        'End_2': row2['End'],
                                        'Strand_2': row2['Strand'],
                                        'Distance': distance,
                                        'Cluster_Size': size
                                    })

            # 根据聚类大小加权计算 ratio
            weighted_sum = sum(seq['Cluster_Size'] for seq in close_sequences)
            ratio = weighted_sum / len(filtered_data) if len(filtered_data) > 0 else 0
            
            # 存储结果
            results.append({
                'Number': int(number),
                'Number_Layer': f'{number}_{layer}',
                'Ratio': ratio
            })

    # 将结果转换为 DataFrame
    distance_result_df = pd.DataFrame(results)
    return distance_result_df


def predict_new_sequences(sequences, model, max_len=50):
    """
    使用模型预测新序列并返回各种评分

    参数:
    sequences (list): 序列列表
    model (sklearn model): 训练好的随机森林模型
    max_len (int): 序列编码的最大长度

    返回:
    dict: 预测结果，包括预测类别、预测概率、预测置信度和特征重要性
    """
    # 序列编码
    encoded_sequences = encode_sequences(sequences, max_len=max_len)
    
    # 手工特征提取
    handcrafted_features = com_seq_feature(sequences)
    
    # 构建特征矩阵
    X = np.hstack([encoded_sequences, handcrafted_features])
    
    # 预测类别
    y_pred = model.predict(X)
    
    # 预测概率
    y_pred_proba = model.predict_proba(X)
    
    # 预测置信度
    y_pred_confidence = np.max(y_pred_proba, axis=1)
    
    # 特征重要性
    feature_importance = model.feature_importances_
    
    return {
        "predictions": y_pred,
        "probabilities": y_pred_proba,
        "confidence": y_pred_confidence,
        "feature_importance": feature_importance
    }

