#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import csv
import pandas as pd
import shutil
from Bio import SeqIO, motifs
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import argparse
import pickle
import sys
from intervaltree import IntervalTree
import glob
from feature import com_seq_feature
from write_to_file import *
from MPI_DFI import *
from analysis_meme_file import *
from model_scoring import *
from gene_interaction import gene_interaction_main  # 导入gene_interaction.py中的主功能
import torch
from Bio import SeqIO
import random
import numpy as np
import matplotlib.pyplot as plt

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

def write_all_motifs_to_one_file(tfbs_df, output_file):
    """
    将所有 motif 写入一个 MEME 格式文件。
    """
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
            write_motif_to_file(pwm_df, output_file, motif_id, append=(not first_write))
            first_write = False
def extract_all_motif_ids(tfbs_df):
    return [f"{num}_{layer}" for num, layer in tfbs_df[['Number','Layer']].drop_duplicates().values]

class MotifAnalyzer:
    def __init__(self, input_file, output_path, model_path='model/tfbs_model.joblib'):
        self.input_file = input_file
        self.output_path = output_path
        self.model_path = model_path

        # 从输入文件名中提取 accession ID
        self.accession = os.path.basename(self.input_file).split('.')[0]


        # 构建预测路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.predict_path = os.path.join(script_dir, '..', 'exclude_GPD_find_key_motif', f'Branch_1_{self.accession}')
        self.predict_path = os.path.normpath(self.predict_path)  # 规范化路径
        if not os.path.exists(self.predict_path):
            print(f"错误: 预测目录不存在: {self.predict_path}")
            sys.exit(1)

        # 定义相关目录
        self.accession_dir = os.path.join(self.output_path, self.accession)
        self.motif_dir = os.path.join(self.accession_dir, 'motif_meme')
        self.fimo_output_dir = os.path.join(self.motif_dir, 'fimo_results')

        # 定义相关文件路径
        self.output_file = os.path.join(self.accession_dir, f"{self.accession}.csv")
        self.meme_file = os.path.join(self.motif_dir, 'final_motif.meme')
        self.fimo_results_tsv = os.path.join(self.fimo_output_dir, 'fimo.txt')
        self.extend_meme_10_path = os.path.join(self.accession_dir, 'extend_meme_10.csv')
        self.results_df_path = os.path.join(self.accession_dir, 'results_df.csv')
        self.rank_list_path = os.path.join(self.accession_dir, 'rank_list.csv')
        self.tomtom_results_path = os.path.join(self.fimo_output_dir, 'tomtom_results')
        self.non_redundant_meme_path = os.path.join(self.motif_dir, 'non_redundant.meme')

        # 定义 MEME Suite 可执行文件路径
        #self.meme_bin_dir = os.path.join(script_dir, '..', 'meme', 'bin')
        #self.fimo_executable = os.path.join(self.meme_bin_dir, 'fimo')
        #self.tomtom_executable = os.path.join(self.meme_bin_dir, 'tomtom')
        
        #self.fimo_executable = shutil.which("fimo")
        #self.tomtom_executable = shutil.which("tomtom")
        self.fimo_executable = '/home/hanzequan/miniconda3/envs/meme-env/bin/fimo'
        self.tomtom_executable = '/home/hanzequan/miniconda3/envs/meme-env/bin/tomtom'
        print('fimo_executable ',self.fimo_executable ,'tomtom_executable',self.tomtom_executable)
        # 验证 MEME Suite 可执行文件是否存在
        if not self.fimo_executable:
            print("警告: 未找到 fimo 可执行文件，请确保它已安装并在 PATH 中可用。")
            sys.exit(1)
        if not self.tomtom_executable:
            print("警告: 未找到 tomtom 可执行文件，请确保它已安装并在 PATH 中可用。")
            sys.exit(1)
        #if not os.path.exists(self.fimo_executable):
         #   print(f"错误: FIMO 可执行文件不存在: {self.fimo_executable}")
        #    sys.exit(1)
        #if not os.path.exists(self.tomtom_executable):
        #    print(f"错误: TOMTOM 可执行文件不存在: {self.tomtom_executable}")
        #    sys.exit(1)

        # 创建必要的目录结构
        self._create_directories()
    def find_input_file(self):
        """
        根据 accession 查找输入文件（.gbk 或 .fasta）。

        :return: 输入文件路径
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(script_dir, '..', 'download_phage', 'phage_gbk')
        gbk_file = os.path.join(download_dir, f"{self.accession}.gbk")
        fasta_file = os.path.join(download_dir, f"{self.accession}.fasta")

        if os.path.exists(gbk_file):
            return gbk_file
        elif os.path.exists(fasta_file):
            return fasta_file
        else:
            print(f"错误: 找不到输入文件 (.gbk 或 .fasta) 对应于 accession: {self.accession}")
            sys.exit(1)

    def _create_directories(self):
        """创建必要的目录结构"""
        os.makedirs(self.accession_dir, exist_ok=True)
        os.makedirs(self.motif_dir, exist_ok=True)
        os.makedirs(self.fimo_output_dir, exist_ok=True)
        print(f"创建目录: {self.accession_dir}")

    def run_analysis(self, model):
        """
        运行完整的分析流程。

        :param model: 预训练的随机森林模型
        :return: max_motif, non_redundant_motif, fimo_results
        """
        # Step 1: 构建 motif 矩阵并过滤
        extend_meme_10 = self.build_and_filter_motifs()

        # Step 2: 移除冗余序列
        rank_list = self.remove_redundant_sequences(extend_meme_10)

        # Step 3: 机器学习打分
        results_df = self.compute_average_probabilities(extend_meme_10, model)

        # Step 4: 距离打分
        distance_score = self.calculate_ratio(extend_meme_10)

        # Step 5: 合并评分数据
        merged_df = self.merge_scores(results_df, distance_score)

        # Step 6: 计算总分和频率分数
        final_df = self.calculate_total_scores(merged_df, rank_list)

        # Step 7: 选择每个 Number 中分数最高的 motif
        max_motif = self.select_max_motif(final_df)

        # Step 8: 绘图
        self.plot_motif_trends(max_motif)

        # Step 9: 选择不同 rank 中分数最高的 motif
        non_redundant_motif = self.select_non_redundant_motif(max_motif)

        # Step 10: 保存结果
        self.save_results(max_motif)

        # Step 11: 生成 MEME 文件
        self.generate_meme_file(non_redundant_motif, extend_meme_10)

        # Step 12: 运行 FIMO
        self.run_fimo_analysis()

        # Step 13: 读取 FIMO 结果
        fimo_results = self.read_fimo_results()

        return max_motif, non_redundant_motif, fimo_results

    def build_and_filter_motifs(self):
        """
        构建 motif 矩阵并进行过滤，如果存在则加载。

        :return: DataFrame，包含所有 motif 信息
        """
        if os.path.exists(self.extend_meme_10_path):
            extend_meme_10 = pd.read_csv(self.extend_meme_10_path)
            print(f"加载已有的 extend_meme_10: {self.extend_meme_10_path}")
        else:
            extend_meme_10 = self.build_motif_matrices(self.predict_path)
            extend_meme_10.to_csv(self.extend_meme_10_path, index=False)
            print(f"保存 extend_meme_10 到: {self.extend_meme_10_path}")
        return extend_meme_10

    def build_motif_matrices(self, directory, sequence_count_occurrences=None):
        """
        构建 motif 矩阵。

        :param directory: 目录路径
        :param sequence_count_occurrences: 字典，用于统计序列出现次数
        :return: DataFrame，包含所有 motif 信息
        """
        if sequence_count_occurrences is None:
            sequence_count_occurrences = {}

        all_motifs_data = pd.DataFrame()
        found_xml = False

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
                                    sequence_count_occurrences[sequence_count] = sequence_count_occurrences.get(sequence_count, 0) + 1
                    except FileNotFoundError:
                        print(f"Warning: Corresponding text file not found: {txt_file}")

                    with open(xml_file) as f:
                        meme_record = motifs.parse(f, "meme")

                    # Read the fasta file
                    if not os.path.exists(fasta_file):
                        print(f"错误: FASTA 文件不存在: {fasta_file}")
                        continue
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
                            suffix = f".{sequence_count_occurrences.get(sequence_count, 1)}" if sequence_count_occurrences.get(sequence_count, 1) > 1 else ""

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

    def remove_redundant_sequences(self, tfbs_df):
        """
        移除冗余序列并保存 rank_list。

        :param tfbs_df: DataFrame，包含 motif 信息
        :return: rank_list DataFrame
        """
        if os.path.exists(self.rank_list_path):
            rank_list = pd.read_csv(self.rank_list_path)
            print(f"加载已有的 rank_list: {self.rank_list_path}")
        else:
            rank_list = self._remove_redundant_sequences_internal(tfbs_df)
            rank_list.to_csv(self.rank_list_path, index=False)
            print(f"保存 rank_list 到: {self.rank_list_path}")
        return rank_list

    def _remove_redundant_sequences_internal(self, tfbs_df):
        """
        内部方法，移除冗余序列。

        :param tfbs_df: DataFrame，包含 motif 信息
        :return: rank_list DataFrame
        """
        # 1. 准备工作环境
        if os.path.exists(self.tomtom_results_path):
            shutil.rmtree(self.tomtom_results_path)
        os.makedirs(self.tomtom_results_path, exist_ok=True)

        # 2. 将所有 motif 合并为一个文件
        combined_motifs_path = os.path.join(self.tomtom_results_path, 'combined_motifs.meme')
        write_all_motifs_to_one_file(tfbs_df, combined_motifs_path)

        # 所有的 motif ID
        all_ids = extract_all_motif_ids(tfbs_df)
        # 3. 全局比对
        tomtom_results = self.run_tomtom_small_e_value(self.tomtom_results_path, combined_motifs_path, combined_motifs_path)
        print('全局比对,tomtom_results',tomtom_results)
        if tomtom_results is None or tomtom_results.empty:
            # 没有匹配，则所有 motif 各自为一cluster, Match_Count=0
            # cluster内就一个motif，不存在相似匹配。每个motif单独成一组。
            rank_df = pd.DataFrame({'Motif_ID': all_ids, 'Match_Count': [0]*len(all_ids)})
            # 单个motif的cluster就是自己，cluster总Match_Count=0，对于排序全一样
            # 为避免并列，可以按Motif_ID进行二次排序
            rank_df = rank_df.sort_values(by=['Match_Count','Motif_ID'], ascending=[False,True]).reset_index(drop=True)
            # 分配群的rank，每个motif是一个独立的cluster
            rank_df['Rank'] = range(1, len(rank_df)+1)
            write_non_redundant_meme(tfbs_df, rank_df['Motif_ID'].tolist(), self.non_redundant_meme_path)
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
        write_non_redundant_meme(tfbs_df, keep_motifs, self.non_redundant_meme_path)

        return rank_df

    def compute_average_probabilities(self, extend_meme_10, model):
        """
        计算每个 Number 和 Layer 的平均概率，并保存 results_df。
    
        :param extend_meme_10: DataFrame，包含 motif 信息
        :param model: 预训练的随机森林模型
        :return: results_df DataFrame
        """
        if os.path.exists(self.results_df_path):
            results_df = pd.read_csv(self.results_df_path)
            print(f"加载已有的 results_df: {self.results_df_path}")
        else:
            # 确保 Number 列为整数
            extend_meme_10['Number'] = extend_meme_10['Number'].astype(int)
            
            # 修改排序方式，直接排序整数
            numbers = sorted(extend_meme_10['Number'].unique())
            results_df = self._compute_average_probabilities_internal(extend_meme_10, model, numbers)
            results_df.to_csv(self.results_df_path, index=False)
            print(f"保存 results_df 到: {self.results_df_path}")
        return results_df

    def _compute_average_probabilities_internal(self, extend_meme_10, model, numbers):
        """
        内部方法，计算平均概率。

        :param extend_meme_10: DataFrame，包含 motif 信息
        :param model: 预训练的随机森林模型
        :param numbers: 要处理的 Number 列表
        :return: results_df DataFrame
        """
        results_list = []
        for number in numbers:
            layers = extend_meme_10[extend_meme_10['Number'] == number]['Layer'].unique()
            for layer in layers:
                sequences = extend_meme_10[
                    (extend_meme_10['Number'] == number) & 
                    (extend_meme_10['Layer'] == layer)
                ]['Sequence'].to_list()
                if not sequences:
                    continue
                results = predict_new_sequences(sequences, model)
                if results is None or "probabilities" not in results:
                    print(f"Warning: 'probabilities' not in results for Number: {number}, Layer: {layer}")
                    continue
                right_column = results["probabilities"][:, 1]
                average_probability = np.mean(right_column)
                results_list.append([number, layer, average_probability])

        results_df = pd.DataFrame(results_list, columns=['Number', 'Layer', 'Average_Probability'])
        results_df['Number'] = results_df['Number'].astype(int)
        results_df = results_df.sort_values(by=['Number', 'Layer'])
        results_df['Number_Layer'] = results_df['Number'].astype(str) + "_" + results_df['Layer']

        return results_df

    def calculate_ratio(self, extend_meme_10):
        """
        计算每个 Number 和 Layer 的距离比率。

        :param extend_meme_10: DataFrame，包含 motif 信息
        :return: distance_result_df DataFrame
        """
        print('计算距离比率...')
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
                dbscan = DBSCAN(eps=50, min_samples=2).fit(positions)
                labels = dbscan.labels_

                # 将聚类结果添加回原数据
                filtered_data = filtered_data.copy()
                filtered_data['Cluster'] = labels

                # 找到接近程度在 50bp 以内的序列对
                close_sequences = []
                for cluster in set(labels):
                    if cluster != -1:
                        cluster_data = filtered_data[filtered_data['Cluster'] == cluster]
                        sequences_in_cluster = cluster_data['Sequence'].tolist()
                        # 生成所有可能的序列对
                        for i in range(len(sequences_in_cluster)):
                            for j in range(i+1, len(sequences_in_cluster)):
                                row1 = cluster_data.iloc[i]
                                row2 = cluster_data.iloc[j]
                                distance = abs(
                                    (row1['End'] if row1['Strand'] == '+' else row1['Start']) - 
                                    (row2['Start'] if row2['Strand'] == '+' else row2['End'])
                                )
                                close_sequences.append((row1['Sequence'], row2['Sequence'], distance))

                # 计算参与近邻关系的序列比例（参与度评分）
                close_sequences_set = set()
                for seq1, seq2, distance in close_sequences:
                    close_sequences_set.add(seq1)
                    close_sequences_set.add(seq2)

                if len(filtered_data) > 0:
                    ratio = len(close_sequences_set) / len(filtered_data)
                else:
                    ratio = 0

                # 存储结果
                results.append({
                    'Number': int(number),
                    'Number_Layer': f'{number}_{layer}',
                    'Ratio': ratio
                })

        # 将结果转换为 DataFrame
        distance_result_df = pd.DataFrame(results)
        return distance_result_df

    def merge_scores(self, results_df, distance_score):
        """
        合并机器学习评分和距离评分。

        :param results_df: DataFrame，包含机器学习评分
        :param distance_score: DataFrame，包含距离评分
        :return: merged_df DataFrame
        """
        results_df = results_df.rename(columns={'Average_Probability': 'Ml_Average_Probability'})
        merged_df = pd.merge(results_df, distance_score, on='Number_Layer', how='inner')
        merged_df = merged_df.rename(columns={'Number_Layer': 'Motif_ID'})
        return merged_df

    def calculate_total_scores(self, merged_df, rank_list):
        """
        计算总分和频率分数。

        :param merged_df: DataFrame，合并后的评分数据
        :param rank_list: DataFrame，rank 信息
        :return: final_df DataFrame
        """
        # 修改列的名称
        merged_df = merged_df.rename(columns={'Number_x': 'Number', 'Number_y': 'Another_Number'})

        final_df = pd.merge(merged_df, rank_list, on='Motif_ID', how='inner')

        # 计算 Total = Ratio + Ml_Average_Probability
        final_df['Total'] = final_df['Ratio'] + final_df['Ml_Average_Probability']

        # 添加频率分数
        rank_counts = final_df['Rank'].value_counts()
        frequent_score_dict = rank_counts / final_df.shape[0]
        final_df['Frequent_score'] = final_df['Rank'].map(frequent_score_dict)
        final_df['Total'] += final_df['Frequent_score']

        # 调整列的位置
        cols = [col for col in final_df.columns if col != 'Total'] + ['Total']
        final_df = final_df[cols]

        return final_df

    def select_max_motif(self, final_df):
        """
        选择每个 Number 中分数最高的 motif。

        :param final_df: DataFrame，包含总分信息
        :return: max_motif DataFrame
        """
        max_motif = final_df.loc[final_df.groupby('Number')['Total'].idxmax()]
        return max_motif

    def plot_motif_trends(self, max_motif):
        """
        绘制 motif 变化趋势图并保存。

        :param max_motif: DataFrame，包含最高分的 motif 信息
        """
        # 确定混乱点：Rank 的计数小于总行数的5%
        rank_counts = max_motif['Rank'].value_counts()
        chaotic_threshold = max_motif.shape[0] / 20  # 5%
        chaotic_ranks = rank_counts[rank_counts < chaotic_threshold].index

        # 为每个唯一的 rank 生成颜色，混乱点使用统一的红色
        unique_ranks = max_motif['Rank'].unique()
        colors = {rank: np.random.rand(3,) for rank in unique_ranks if rank not in chaotic_ranks}
        chaotic_color = 'red'  # 混乱点的颜色

        # 创建绘图
        plt.figure(figsize=(28, 6))
        plt.plot(max_motif['Number'], max_motif['Total'], linestyle='-', color='gray', linewidth=1, label='Trend Line')

        # 绘制每个点
        for rank in unique_ranks:
            rank_data = max_motif[max_motif['Rank'] == rank]
            if rank in chaotic_ranks:
                plt.scatter(rank_data['Number'], rank_data['Total'], color=chaotic_color, 
                            label='Chaotic Point' if 'Chaotic Point' not in plt.gca().get_legend_handles_labels()[1] else "", s=100)
            else:
                plt.scatter(rank_data['Number'], rank_data['Total'], color=colors[rank], 
                            label=f'Rank {rank}', s=100)

        # 自定义图表
        plt.xlabel('Number', fontsize=14)
        plt.ylabel('Total', fontsize=14)
        plt.title('Total vs Number with Connected Points and Chaotic Points', fontsize=16)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=12)
        plt.legend(title="Rank", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        # 保存图表
        plot_path = os.path.join(self.accession_dir, 'motif_trends.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"保存图表到: {plot_path}")

    def select_non_redundant_motif(self, max_motif):
        """
        选择不同 rank 中分数最高的 motif。

        :param max_motif: DataFrame，包含最高分的 motif 信息
        :return: non_redundant_motif DataFrame
        """
        rank_threshold = max_motif.shape[0] / 20  # 5%
        valid_ranks = max_motif['Rank'].value_counts()
        valid_ranks = valid_ranks[valid_ranks > rank_threshold].index
        non_redundant_motif = max_motif[max_motif['Rank'].isin(valid_ranks)]
        non_redundant_motif = non_redundant_motif.loc[non_redundant_motif.groupby('Rank')['Total'].idxmax()]
        non_redundant_motif = non_redundant_motif.reset_index(drop=True)
        return non_redundant_motif

    def save_results(self, max_motif):
        """
        保存分析结果到 CSV 文件。

        :param max_motif: DataFrame，包含最高分的 motif 信息
        """
        max_motif['Accession'] = self.accession
        max_motif.to_csv(self.output_file, index=False)
        print(f"保存结果到: {self.output_file}")

    def generate_meme_file(self, non_redundant_motif, extend_meme_10):
        """
        生成 MEME 文件。
    
        :param non_redundant_motif: DataFrame，包含非冗余的 motif 信息
        :param extend_meme_10: DataFrame，包含所有 motif 信息
        """
        # 提取需要的列
        non_redundant_motif_subset = non_redundant_motif[['Number', 'Layer']].copy()
        extend_meme_10_subset = extend_meme_10[['Number', 'Layer', 'Strand', 'Start', 'End', 
                                                 'Start_extend', 'End_extend', 'p-value', 'e-value', 
                                                 'Sequence', 'Motif', 'Extend_sequence']].copy()
    
        # 确保 'Number' 列为整数类型
        non_redundant_motif_subset['Number'] = non_redundant_motif_subset['Number'].astype(int)
        extend_meme_10_subset['Number'] = extend_meme_10_subset['Number'].astype(int)
    
        # 合并数据
        non_redundant_sequence = pd.merge(non_redundant_motif_subset, extend_meme_10_subset, on=['Number', 'Layer'], how='inner')
    
        # 生成 MEME 文件
        generate_all_pwm_and_write_to_meme(non_redundant_sequence, self.motif_dir)
        print(f"生成 MEME 文件到: {self.meme_file}")

    def run_fimo_analysis(self):
        """
        运行 FIMO 分析，包括转换 GenBank 文件为 FASTA 格式并运行 FIMO。
        """
        self.convert_and_run_fimo(
            motif_file=self.meme_file,
            output_dir=self.fimo_output_dir
        )

    def convert_and_run_fimo(self, motif_file, output_dir):
        """
        转换 GenBank 文件为 FASTA 并运行 FIMO。

        :param motif_file: motif 文件路径（MEME 格式）
        :param output_dir: FIMO 搜索结果的输出目录
        """
        # 确定输入文件类型
        if self.input_file.lower().endswith('.gbk') or self.input_file.lower().endswith('.gb'):
            # 如果是 GenBank 文件，解析为 FASTA
            output_fasta = os.path.splitext(self.input_file)[0] + ".fasta"
            self.parse_genbank_to_fasta(self.input_file, output_fasta)
        elif self.input_file.lower().endswith('.fasta') or self.input_file.lower().endswith('.fa'):
            output_fasta = self.input_file
        else:
            print(f"错误: 输入文件必须是 GenBank (.gbk/.gb) 或 FASTA (.fasta/.fa) 格式。")
            return

        # 使用 FIMO 进行 motif 搜索，设置阈值为 1e-6
        fimo_success = self.run_fimo(motif_file, output_fasta, output_dir, threshold=1e-3)

        if fimo_success:
            # FIMO 成功，检查 fimo.tsv 文件是否存在
            fimo_tsv_path = os.path.join(output_dir, "fimo.tsv")
            fimo_tsv_output_path = os.path.join(output_dir, "fimo_results.tsv")  # 输出路径

            if os.path.exists(fimo_tsv_path):
                self.convert_fimo_tsv_to_tsv(fimo_tsv_path, fimo_tsv_output_path)
                print(f"FIMO 结果已转换为 TSV 格式: {fimo_tsv_output_path}")
            else:
                print(f"错误: {fimo_tsv_path} 不存在。")
        else:
            print("FIMO 未成功运行，跳过转换。")

    def parse_genbank_to_fasta(self, gbk_file, output_fasta):
        """
        将 GenBank 文件转换为 FASTA 格式。

        :param gbk_file: GenBank 文件路径
        :param output_fasta: 输出的 FASTA 文件路径
        """
        if not os.path.exists(gbk_file):
            print(f"错误: GenBank 文件不存在: {gbk_file}")
            return

        with open(output_fasta, 'w') as fasta_file:
            for record in SeqIO.parse(gbk_file, "genbank"):
                fasta_file.write(f">{record.id}\n")
                fasta_file.write(str(record.seq) + "\n")
        print(f"已将 GenBank 文件转换为 FASTA 格式: {output_fasta}")

    def run_fimo(self, motif_file, genome_fasta, output_dir, threshold=1e-3):
        """
        运行 FIMO 并捕获输出。

        :param motif_file: motif 文件路径（MEME 格式）
        :param genome_fasta: 基因组 FASTA 文件路径
        :param output_dir: FIMO 搜索结果的输出目录
        :param threshold: FIMO 匹配阈值
        :return: 是否成功运行 FIMO
        """
        if not os.path.exists(motif_file):
            print(f"错误: Motif 文件未找到: {motif_file}")
            return False
        if not os.path.exists(genome_fasta):
            print(f"错误: 基因组 FASTA 文件未找到: {genome_fasta}")
            return False

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建 FIMO 输出目录: {output_dir}")

        # 设置 FIMO 命令，使用绝对路径
        fimo_cmd = [
            self.fimo_executable,  # FIMO 的绝对路径
            "--oc", output_dir,    # 输出目录
            "--thresh", str(threshold),  # 阈值
            motif_file,            # motif 文件路径
            genome_fasta           # 基因组 FASTA 文件路径
        ]

        print(f"运行 FIMO 命令: {' '.join(fimo_cmd)}")

        try:
            result = subprocess.run(fimo_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"FIMO 搜索成功完成。标准输出:\n{result.stdout}")
            if result.stderr:
                print(f"FIMO 标准错误输出:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"运行 FIMO 时发生错误: {e}")
            print(f"标准输出:\n{e.stdout}")
            print(f"标准错误输出:\n{e.stderr}")
            return False

        return True

    def convert_fimo_tsv_to_tsv(self, fimo_tsv_path, fimo_tsv_output_path):
        """
        将 FIMO 的 TSV 输出文件转换为新的 TSV 文件。

        :param fimo_tsv_path: 原始 FIMO TSV 文件路径
        :param fimo_tsv_output_path: 转换后的 TSV 文件路径
        """
        with open(fimo_tsv_path, 'r') as fimo_file:
            lines = fimo_file.readlines()

        # 解析并提取匹配的信息
        results = []
        for line in lines:
            if line.startswith("#") or line.strip() == "":
                continue  # 跳过注释和空行
            if line.startswith("motif"):
                # 这是标题行，保持不变
                header = line.strip().split('\t')
                results.append(header)
                continue
            columns = line.strip().split('\t')
            if len(columns) >= 9:  # 确保行有足够的列
                results.append(columns)

        # 将提取的结果写入 TSV 文件
        with open(fimo_tsv_output_path, 'w', newline='') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            for row in results:
                tsv_writer.writerow(row)

        print(f"已将 FIMO 的 TSV 输出文件转换为: {fimo_tsv_output_path}")


    def read_fimo_results(self):
        """
        读取 FIMO 结果。
    
        :return: FIMO 结果的 DataFrame 或 None
        """
        try:
            # 定义新的列名
            column_names = [
                'motif_id', 'sequence_name', 'start', 'stop', 
                'strand', 'score', 'p-value', 'q-value', 'matched_sequence'
            ]
            
            # 读取 FIMO 结果文件，并使用新的列名
            fimo_results = pd.read_csv(self.fimo_results_tsv, sep='\t', names=column_names, header=None)
            
            print(f"成功读取 FIMO 结果: {self.fimo_results_tsv}")
            return fimo_results
        except FileNotFoundError:
            print(f"错误: FIMO 结果文件不存在: {self.fimo_results_tsv}")
            return None
        except Exception as e:
            print(f"读取 FIMO 结果时发生错误: {e}")
            return None
    def run_tomtom_small_e_value(self, result_path, query_motif, target_motif):
        """
        运行 Tomtom 分析，并返回结果。

        :param result_path: Tomtom 结果输出目录
        :param query_motif: 查询 motif 文件路径
        :param target_motif: 目标 motif 文件路径
        :return: Tomtom 结果 DataFrame 或 None
        """
        tomtom_command = [
            self.tomtom_executable,  # Tomtom 的绝对路径
            "-dist", "ed",            # 使用欧氏距离作为相似性度量，更严格
            "-min-overlap", "8",      # 要求至少8个碱基位点的重叠
            "-thresh", "1e-10",       # 更低的阈值，更严格的筛选
            "-oc", result_path,
            query_motif,
            target_motif,
            '-verbosity', '1'
        ]
        print('tomtom_command',tomtom_command)
        try:
            subprocess.run(tomtom_command, check=True)
            print("Tomtom 分析成功完成。")
        #except subprocess.CalledProcessError as e:
        #    print(f"运行 Tomtom 时发生错误: {e}")
        #    return None
        except subprocess.CalledProcessError as e:
            print(f"运行 Tomtom 时发生错误: {e}")
            
            # 打印标准错误信息，如果 stderr 为 None 则避免调用 decode()
            if e.stderr:
                print("标准错误:", e.stderr.decode())
            else:
                print("没有标准错误输出")
            
            # 打印标准输出，如果 stdout 为 None 则避免调用 decode()
            if e.stdout:
                print("标准输出:", e.stdout.decode())
            else:
                print("没有标准输出")
            
            return None
        output_file = os.path.join(result_path, 'tomtom.txt')
        print('tomtom output_file',output_file)
        return self.read_tomtom_results(output_file)

    def read_tomtom_results(self, output_file):
        """
        读取 Tomtom 分析的结果。
    
        :param output_file: Tomtom 结果文件路径
        :return: Tomtom 结果 DataFrame 或 None
        """
        try:
            # 更新列名，确保读取的文件列名和文件内容一致
            column_names = [
                "Query_ID", "Target_ID", "Optimal_offset", "p-value", "E-value", 
                "q-value", "Overlap", "Query_consensus", "Target_consensus", "Orientation"
            ]
        
            # 读取文件并指定列名，header=None 不读取文件中的第一行作为列名
            tomtom_results = pd.read_csv(output_file, sep='\t', names=column_names, header=None)
        
            print(f"成功读取 Tomtom 结果: {output_file}")
            return tomtom_results
        
        except FileNotFoundError:
            print(f"错误: Tomtom 结果文件不存在: {output_file}")
            return None
        except Exception as e:
            print(f"读取 Tomtom 结果时发生错误: {e}")
            return None
def generate_all_pwm_and_write_to_meme(tfbs_df, output_dir):
    """
    将给定的 motif 数据框写入 MEME 格式文件。

    :param tfbs_df: 包含 motif 信息的 DataFrame
    :param output_dir: MEME 文件的输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    meme_output_path = os.path.join(output_dir, 'final_motif.meme')

    # 初始化文件，写入头部信息
    with open(meme_output_path, 'w') as file:
        file.write('MEME version 4\n\n')
        file.write('ALPHABET= ACGT\n\n')
        file.write('strands: + -\n\n')
        file.write('Background letter frequencies\n')
        file.write('A 0.25 C 0.25 G 0.25 T 0.25\n\n')

    unique_numbers = tfbs_df['Number'].unique()
    for number in unique_numbers:
        df_number = tfbs_df[tfbs_df['Number'] == number]
        unique_layers = df_number['Layer'].unique()

        for idx, layer in enumerate(unique_layers):
            df_layer = df_number[df_number['Layer'] == layer]
            sequences = [Seq(seq) for seq in df_layer['Sequence']]
            m = motifs.create(sequences)

            pwm = m.counts.normalize(pseudocounts={'A': 0, 'C': 0, 'G': 0, 'T': 0})
            pwm_df = pd.DataFrame({nucleotide: pwm[nucleotide] for nucleotide in 'ACGT'})

            if not isinstance(pwm_df, pd.DataFrame):
                raise TypeError(f"Expected pwm_df to be a pandas DataFrame, but got {type(pwm_df)}")

            write_motif_to_file_old(pwm_df, meme_output_path, layer, number, append=True)

    print(f"生成 MEME 文件到: {meme_output_path}")
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
def write_motif_to_file(pwm_df, file_path, motif_id,  append=True):
    """
    将给定的 PWM 矩阵写入 MEME 格式文件。

    :param pwm_df: 包含 PWM 信息的 DataFrame
    :param file_path: MEME 文件路径
    :param motif_name: motif 的名称
    :param number: motif 的编号
    :param append: 是否以追加模式写入
    """
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
def merge_sequence_contexts(directory_path, file_prefixes):
    """
    合并指定目录下多个CSV文件中的序列上下文，并记录每个合并区间内的最高interaction_strength。

    参数:
    - directory_path (str): 存放CSV文件的根目录路径。
    - file_prefixes (list of str): 需要处理的文件前缀列表。

    返回:
    - pd.DataFrame: 合并后的DataFrame，包含 'accession', 'start', 'end', 'sequence', 'score' 列。
    """
    # 用于存储最终合并结果的列表
    large_language_tfbs = []

    # 依次处理每个前缀
    for prefix in file_prefixes:
        # 获取当前前缀对应的目录
        prefix_directory = os.path.join(directory_path, prefix)

        # 获取所有 CSV 文件
        csv_files = glob.glob(os.path.join(prefix_directory, "*.csv"))

        if not csv_files:
            print(f"No CSV files found in directory: {prefix_directory}")
            continue

        # 合并所有 CSV
        combined_df = pd.concat(
            (pd.read_csv(file) for file in csv_files),
            ignore_index=True
        )

        # 添加 accession 列
        combined_df['accession'] = prefix

        # 将 absolute_genomic_position 转为 int
        combined_df['absolute_genomic_position'] = combined_df['absolute_genomic_position'].astype(int)

        # 去重：同一个 absolute_genomic_position 只保留第一行
        unique_rows = combined_df.drop_duplicates(
            subset=['absolute_genomic_position'],
            keep='first'
        )

        # 增加合并区间的起止列
        unique_rows['absolute_start'] = unique_rows['absolute_genomic_position'] - 10
        unique_rows['absolute_end'] = unique_rows['absolute_genomic_position'] + 10

        # 按照 absolute_start 升序排序
        unique_rows = unique_rows.sort_values(by='absolute_start', ascending=True).reset_index(drop=True)

        # 用于存储当前前缀合并结果的列表
        merged_data = []

        i = 0
        while i < len(unique_rows):
            current_row = unique_rows.iloc[i]

            # 当前合并段的起止坐标
            min_start = current_row['absolute_start']
            max_end = current_row['absolute_end']

            # 当前合并段的 sequence（先放第一个行的序列）
            merged_sequence = current_row['sequence_context']

            # 记录所有被合并区间中最高的 interaction_strength
            max_score = current_row['interaction_strength']

            j = i + 1
            # 当后续行与当前段重叠，就进行合并
            while j < len(unique_rows) and unique_rows.iloc[j]['absolute_start'] <= max_end:
                next_row = unique_rows.iloc[j]

                # 计算重叠长度，基于当前的 max_end
                overlap_length = max_end - next_row['absolute_start'] + 1

                # 确保 overlap_length 不超过下一个 'sequence_context' 的长度
                if overlap_length > len(next_row['sequence_context']):
                    overlap_length = len(next_row['sequence_context'])

                # 如果有重叠，仅添加非重叠部分的下一个序列
                if overlap_length > 0:
                    merged_sequence += next_row['sequence_context'][overlap_length:]
                else:
                    merged_sequence += next_row['sequence_context']

                # 更新最高 interaction_strength
                if next_row['interaction_strength'] > max_score:
                    max_score = next_row['interaction_strength']

                # 在计算 overlap_length 之后再更新 max_end
                max_end = max(max_end, next_row['absolute_end'])

                j += 1

            # 把合并好的区间及其最高分数记录下来
            merged_data.append({
                'accession': prefix,
                'start': min_start,
                'end': max_end,
                'sequence': merged_sequence,
                'score': max_score
            })

            # 跳过已处理的重叠部分，继续下一个不重叠的区间
            i = j

        # 把当前前缀的合并结果转为 DataFrame 后存入列表
        merged_df = pd.DataFrame(merged_data)
        large_language_tfbs.append(merged_df)

    # 如果所有前缀都没有数据，返回空 DataFrame
    if not large_language_tfbs:
        print("No data to combine across all prefixes.")
        return pd.DataFrame()

    # 将所有前缀的结果合并
    final_df = pd.concat(large_language_tfbs, ignore_index=True)

    return final_df
def main():
    from joblib import load
    parser = argparse.ArgumentParser(description='Motif Analyzer')
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Input GenBank or FASTA file path (e.g., /path/to/NC_005856.gbk)')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output directory path to store results')
    
    # 新增参数
    parser.add_argument('--pfam_hmm', type=str, default=os.path.join( 'Pfam', 'Pfam-A.hmm'),
                        help='Path to Pfam HMM database file (default: scripts/Pfam/Pfam-A.hmm)')
    parser.add_argument('--model_path', type=str, default=os.path.join( 'model', 'megaDNA_phage_145M.pt'),
                        help='Path to the model file (default: model/megaDNA_phage_145M.pt)')
    parser.add_argument('--rf_model_path', type=str, default=os.path.join( 'model', 'tfbs_model.joblib'),
                        help='Path to the random forest model file (default: model/tfbs_model.joblib)')
    
    args = parser.parse_args()

    # 检查 Pfam HMM 文件
    if not os.path.exists(args.pfam_hmm):
        print(f"错误: Pfam HMM 文件不存在: {args.pfam_hmm}")
        sys.exit(1)

    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)

    # 检查随机森林模型文件
    if not os.path.exists(args.rf_model_path):
        print(f"错误: 随机森林模型文件不存在: {args.rf_model_path}")
        sys.exit(1)

    # 加载随机森林模型
    rf = load(args.rf_model_path)
    print(f"加载随机森林模型: {args.rf_model_path}")

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    interaction_output_dir = os.path.join(args.output_path, 'interaction')
    os.makedirs(interaction_output_dir, exist_ok=True)

    # 整合 gene_interaction.py 的功能
    if os.path.isfile(args.input_file):
        temp_input_dir = os.path.join(args.output_path, 'temp_gbk')
        os.makedirs(temp_input_dir, exist_ok=True)
        shutil.copy(args.input_file, temp_input_dir)
        input_dir = temp_input_dir
    elif os.path.isdir(args.input_file):
        input_dir = args.input_file
    else:
        print(f"输入路径 {args.input_file} 无效。")
        sys.exit(1)

    # 初始化模型和设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载主要模型
    model = torch.load(args.model_path, map_location=torch.device(device))
    print(f"加载模型: {args.model_path} 到设备: {device}")

    # 调用 gene_interaction.py 的功能
    try:
        gene_interaction_main(
            input_dir=input_dir,
            output_dir=interaction_output_dir,
            pfam_hmm=args.pfam_hmm,
            protein_names=["Repressor", "Inhibitor", "ProteinX"],  # 替换为您感兴趣的实际蛋白质名称
            exclude_keywords=["Cro", "anti", "cro", "CRO"],
            model=model,
            device=device
        )
        print("基因互作分析完成。")
    except NameError:
        print("错误: gene_interaction_main 未定义。请确保已正确导入 gene_interaction_main 函数。")
        sys.exit(1)

    # 创建 MotifAnalyzer 实例
    try:
        analyzer = MotifAnalyzer(
            input_file=args.input_file,
            output_path=args.output_path,
            model_path=args.rf_model_path
        )
    except NameError:
        print("错误: MotifAnalyzer 未定义。请确保已正确导入 MotifAnalyzer 类。")
        sys.exit(1)

    # 运行分析
    if rf is not None:
        max_motif, non_redundant_motif, fimo_results = analyzer.run_analysis(rf)

        if fimo_results is not None:
            print(f"FIMO 结果预览:\n{fimo_results.head()}")

    # -------------------- 新增的处理步骤开始 --------------------

    # 定义变量
    directory_path = os.path.join(interaction_output_dir, 'batch_figure')
    accession = os.path.splitext(os.path.basename(args.input_file))[0]
    file_prefixes = [accession]

    # 调用 merge_sequence_contexts 函数
    llm_matrix = merge_sequence_contexts(directory_path, file_prefixes)
    if llm_matrix.empty:
        print("llm_matrix 为空，无法继续后续处理。")
        sys.exit(1)
    else:
        print("llm_matrix 生成成功。")

    # 读取 motif_meme/fimo_results/fimo.txt
    fimo_path = os.path.join(args.output_path, accession, 'motif_meme', 'fimo_results', 'fimo.txt')
    if not os.path.exists(fimo_path):
        print(f"错误: FIMO 结果文件不存在: {fimo_path}")
        sys.exit(1)
    motif_metrix = pd.read_csv(fimo_path, sep='\t')
    print(f"FIMO 文件 {fimo_path} 读取成功。")

    # 读取 output_path/{accession}/{accession}.csv
    motif_score_path = os.path.join(args.output_path, accession, f"{accession}.csv")
    if not os.path.exists(motif_score_path):
        print(f"错误: Motif score 文件不存在: {motif_score_path}")
        sys.exit(1)
    motif_score = pd.read_csv(motif_score_path)
    print(f"Motif score 文件 {motif_score_path} 读取成功。")

    # 获取唯一的 motif patterns
    unique_patterns = motif_metrix['#pattern name'].unique()

    # 过滤 motif_score 并选择正确的列
    required_motif_score_columns = {'Motif_ID', 'Total'}
    if not required_motif_score_columns.issubset(motif_score.columns):
        print(f"错误: motif_score 文件缺少必要的列: {required_motif_score_columns - set(motif_score.columns)}")
        sys.exit(1)
    filtered_motif_score = motif_score[motif_score['Motif_ID'].isin(unique_patterns)][['Motif_ID', 'Total']]

    # 重命名 'Total' 列为 'evolution_score'
    filtered_motif_score = filtered_motif_score.rename(columns={'Total': 'evolution_score'})

    # 合并到 motif_metrix
    motif_metrix = motif_metrix.merge(
        filtered_motif_score,
        left_on='#pattern name',
        right_on='Motif_ID',
        how='left'
    )

    # 删除不需要的 'Motif_ID' 列
    motif_metrix = motif_metrix.drop(columns=['Motif_ID'])

    # 确保 start 和 stop 列是整数类型
    try:
        motif_metrix['start'] = motif_metrix['start'].astype(int)
        motif_metrix['stop'] = motif_metrix['stop'].astype(int)
    except KeyError as e:
        print(f"错误: motif_metrix 缺少列 {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"错误: 无法将 'start' 或 'stop' 列转换为整数: {e}")
        sys.exit(1)

    # 检查 llm_matrix 的列名
    print("Columns in llm_matrix:")
    print(llm_matrix.columns)

    # 确保 llm_matrix 包含 'start', 'end' 和 'score' 列
    required_columns = {'start', 'end', 'score'}
    if not required_columns.issubset(llm_matrix.columns):
        print(f"错误: llm_matrix 缺少必要的列: {required_columns - set(llm_matrix.columns)}")
        sys.exit(1)

    try:
        llm_matrix['start'] = llm_matrix['start'].astype(int)
        llm_matrix['end'] = llm_matrix['end'].astype(int)
        llm_matrix['score'] = llm_matrix['score'].astype(float)  # 根据实际情况调整类型
    except ValueError as e:
        print(f"错误: 无法转换 llm_matrix 的列类型: {e}")
        sys.exit(1)

    # 创建 IntervalTree
    tree = IntervalTree()
    for _, row in llm_matrix.iterrows():
        # IntervalTree 的区间是左闭右开，所以 end 需要加 1
        tree[row['start']:row['end']+1] = row['score']

    # 定义函数获取 llm_score，取最大值
    def get_llm_score(motif_row):
        overlapping = tree[motif_row['start']:motif_row['stop']+1]
        if overlapping:
            # 取所有重叠的 score 中的最大值
            return max(interval.data for interval in overlapping)
        else:
            return 0  # 或者使用 None，根据需求

    # 应用函数生成 'llm_score' 列
    motif_metrix['llm_score'] = motif_metrix.apply(get_llm_score, axis=1)
    print("llm_score 计算完成。")

    # 最后保存 motif_metrix
    output_motif_metrix_path = os.path.join(args.output_path, accession, f"{accession}_motif_metrix.csv")
    os.makedirs(os.path.dirname(output_motif_metrix_path), exist_ok=True)  # 确保目录存在
    motif_metrix.to_csv(output_motif_metrix_path, index=False)
    print(f"motif_metrix 保存成功: {output_motif_metrix_path}")

    # -------------------- 新增的处理步骤结束 --------------------

if __name__ == "__main__":
    main()
