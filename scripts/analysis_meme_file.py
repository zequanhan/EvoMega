from Bio.Seq import Seq
import shutil
import subprocess

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
from write_to_file import *
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
def remove_redundant_sequences(tfbs_df, 
                              output_dir='/home/hanzequan/pwm_files', 
                              result_path='/home/hanzequan/pwm_files/tomtom_results',
                              final_output='/home/hanzequan/pwm_files/non_redundant.meme'):
    # 1. 准备工作环境
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    # 2. 将所有 motif 合并为一个文件
    combined_motifs_path = os.path.join(result_path, 'combined_motifs.meme')
    write_all_motifs_to_one_file(tfbs_df, combined_motifs_path)

    # 所有的 motif ID
    all_ids = extract_all_motif_ids(tfbs_df)

    # 3. 全局比对
    tomtom_results = run_tomtom_small_e_value(result_path, combined_motifs_path, combined_motifs_path)
    if tomtom_results is None or tomtom_results.empty:
        # 没有匹配，则所有 motif 各自为一cluster, Match_Count=0
        # cluster内就一个motif，不存在相似匹配。每个motif单独成一组。
        rank_df = pd.DataFrame({'Motif_ID': all_ids, 'Match_Count': [0]*len(all_ids)})
        # 单个motif的cluster就是自己，cluster总Match_Count=0，对于排序全一样
        # 为避免并列，可以按Motif_ID进行二次排序
        rank_df = rank_df.sort_values(by=['Match_Count','Motif_ID'], ascending=[False,True]).reset_index(drop=True)
        # 分配群的rank，每个motif是一个独立的cluster
        rank_df['Rank'] = range(1, len(rank_df)+1)
        write_non_redundant_meme(tfbs_df, rank_df['Motif_ID'].tolist(), final_output)
        return rank_df

    # 去除自匹配行（如果不需要可注释掉）
    tomtom_results = tomtom_results[tomtom_results['Query_ID'] != tomtom_results['Target_ID']]

    # 确保 tomtom_results 中的 ID 都在 all_ids 中
    tomtom_results = tomtom_results[tomtom_results['Query_ID'].isin(all_ids) & tomtom_results['Target_ID'].isin(all_ids)]

    # 计算每个motif的Match_Count（作为Query匹配的数量）
    match_counts = tomtom_results['Query_ID'].value_counts().to_dict()
    for m in all_ids:
        if m not in match_counts:
            match_counts[m] = 0

    # 4. 使用并查集，将相似的motif聚合成 cluster
    parent = {m: m for m in all_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    # 构建匹配关系用于合并
    # 每对 (Query_ID, Target_ID) 表示这两个motif相似，进行union操作
    for _, row in tomtom_results.iterrows():
        q, t = row['Query_ID'], row['Target_ID']
        union(q, t)

    # 找出所有cluster
    clusters = {}
    for m in all_ids:
        r = find(m)
        if r not in clusters:
            clusters[r] = []
        clusters[r].append(m)

    # 5. 对cluster进行评分：定义"motif相似的最多"为 cluster内所有 motif 的Match_Count之和越大，越相似
    cluster_info = []
    for root, members in clusters.items():
        # 计算cluster总的Match_Count
        cluster_sum = sum(match_counts[m] for m in members)
        # 同时记录cluster大小或root，用于二次排序避免并列
        cluster_info.append((root, cluster_sum, len(members)))

    # 对cluster排序：按总Match_Count降序，如果有相同则按cluster大小降序，再按root字典序排序
    # 确保没有并列
    cluster_info.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)

    # 分配cluster rank
    # rank=1表示总Match_Count最大的那个cluster
    cluster_rank_map = {}
    current_rank = 1
    for c in cluster_info:
        root = c[0]
        cluster_rank_map[root] = current_rank
        current_rank += 1

    # 为每个motif分配rank，motif的rank就是其所在cluster的rank
    motif_rank_list = []
    for root, members in clusters.items():
        cr = cluster_rank_map[root]
        for m in members:
            motif_rank_list.append((m, match_counts[m], cr))

    rank_df = pd.DataFrame(motif_rank_list, columns=['Motif_ID','Match_Count','Rank'])

    # 同一cluster内motif有相同的Rank，实现"将所有相似motif聚集到一起，然后根据cluster总体相似度排名"
    # 一个rank对应一个cluster，不会有多个cluster共享rank
    # 一个rank中有多个motif是因为一个cluster本身包含多个相似motif，但它们是同一个group

    # 写出non_redundant meme
    # 这里全部保留，如有需要可在后续进行过滤
    keep_motifs = rank_df['Motif_ID'].tolist()
    write_non_redundant_meme(tfbs_df, keep_motifs, final_output)

    return rank_df
def run_tomtom_small_e_value(result_path, query_motif, target_motif):
    tomtom_command = [
        "/home/hanzequan/meme/bin/tomtom",
        "-dist", "ed",            # 使用欧氏距离作为相似性度量，更严格
        "-min-overlap", "8",      # 要求至少8个碱基位点的重叠
        "-thresh", "1e-10",       # 更低的阈值，更严格的筛选
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
