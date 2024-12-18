import os
import subprocess
import csv
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
from joblib import load

from feature import com_seq_feature
from write_to_file import *
from MPI_DFI import *
from analysis_meme_file import *
from model_scoring import *

class MotifAnalyzer:
    def __init__(self, accession, base_output_dir='/home/hanzequan/test_bectiral/operator_recongize/picture_10_phage_tfbs/EvoMega/new_score_system'):
        self.accession = accession
        self.predict_path = f'/home/hanzequan/test_bectiral/operator_recongize/exclude_GPD_find_key_motif/Branch_1_{accession}'
        self.base_output_dir = base_output_dir
        self.accession_dir = os.path.join(self.base_output_dir, accession)
        self.motif_dir = os.path.join(self.accession_dir, 'motif_meme')
        self.fimo_output_dir = os.path.join(self.motif_dir, 'fimo_results')
        self.output_file = os.path.join(self.accession_dir, f"{accession}.csv")
        self.meme_file = os.path.join(self.motif_dir, 'final_motif.meme')
        self.fimo_results_tsv = os.path.join(self.fimo_output_dir, 'fimo_results.tsv')
        self.extend_meme_10_path = os.path.join(self.accession_dir, 'extend_meme_10.csv')
        self.results_df_path = os.path.join(self.accession_dir, 'results_df.csv')
        self.rank_list_path = os.path.join(self.accession_dir, 'rank_list.csv')
    
        # 创建目录
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录结构"""
        os.makedirs(self.motif_dir, exist_ok=True)
        os.makedirs(self.fimo_output_dir, exist_ok=True)
        print(f"创建目录: {self.accession_dir}")

    def run_analysis(self, rf):
        """运行完整的分析流程"""
        # Step 1: 构建 motif 矩阵并过滤
        extend_meme_10 = self.build_and_filter_motifs()

        # Step 2: 移除冗余序列
        rank_list = self.remove_redundant_sequences(extend_meme_10)

        # Step 3: 机器学习打分
        results_df = self.compute_average_probabilities(extend_meme_10, rf)

        # Step 4: 距离打分
        distance_score = calculate_ratio(extend_meme_10)

        # Step 5: 合并评分数据
        merged_df = self.merge_scores(results_df, distance_score)

        # Step 6: 计算总分和频率分数
        final_df, Total_score = self.calculate_total_scores(merged_df, rank_list)

        # Step 7: 选择每个 Number_x 中分数最高的 motif
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
        """构建 motif 矩阵并进行过滤，如果存在则加载"""
        if os.path.exists(self.extend_meme_10_path):
            extend_meme_10 = pd.read_csv(self.extend_meme_10_path)
            print(f"加载已有的 extend_meme_10: {self.extend_meme_10_path}")
        else:
            extend_meme_10 = build_motif_matrices(self.predict_path)
            # 如果需要对 extend_meme_10 进行过滤，可以在这里取消注释并调整
            # 针对NC_002371进行过滤
            extend_meme_10 = extend_meme_10[
                (extend_meme_10['Number'].astype(int) >= 1) &
                (extend_meme_10['Number'].astype(int) <= 101)
            ]
            extend_meme_10.to_csv(self.extend_meme_10_path, index=False)
            print(f"保存 extend_meme_10 到: {self.extend_meme_10_path}")
        return extend_meme_10

    def remove_redundant_sequences(self, extend_meme_10):
        """移除冗余序列并保存 rank_list"""
        if os.path.exists(self.rank_list_path):
            rank_list = pd.read_csv(self.rank_list_path)
            print(f"加载已有的 rank_list: {self.rank_list_path}")
        else:
            rank_list = remove_redundant_sequences(extend_meme_10)
            rank_list.to_csv(self.rank_list_path, index=False)
            print(f"保存 rank_list 到: {self.rank_list_path}")
        return rank_list

    def compute_average_probabilities(self, extend_meme_10, rf):
        """计算机器学习评分并保存 results_df"""
        if os.path.exists(self.results_df_path):
            results_df = pd.read_csv(self.results_df_path)
            print(f"加载已有的 results_df: {self.results_df_path}")
        else:
            numbers = sorted(extend_meme_10['Number'].unique(), key=int)
            results_df = compute_average_probabilities(extend_meme_10, rf, numbers)
            results_df.to_csv(self.results_df_path, index=False)
            print(f"保存 results_df 到: {self.results_df_path}")
        return results_df

    def merge_scores(self, results_df, distance_score):
        """合并机器学习评分和距离评分"""
        results_df = results_df.rename(columns={'Average_Probability': 'Ml_Average_Probability'})
        merged_df = pd.merge(results_df, distance_score, left_on='Number_Layer', right_on='Number_Layer', how='inner')
        merged_df = merged_df.rename(columns={'Number_Layer': 'Motif_ID'})
        return merged_df

    def calculate_total_scores(self, merged_df, rank_list):
        """计算总分和频率分数"""
        merged_df = merged_df.rename(columns={'Number_x': 'Number', 'Number_y': 'Another_Number'})
        final_df = pd.merge(merged_df, rank_list, left_on='Motif_ID', right_on='Motif_ID', how='inner')

        # 计算 Total = Ratio + Ml_Average_Probability
        final_df['Total'] = final_df['Ratio'] + final_df['Ml_Average_Probability']

        # 添加频率分数
        rank_counts = final_df['Rank'].value_counts()
        frequent_score_dict = rank_counts / final_df.shape[0]
        final_df['Frequent_score'] = final_df['Rank'].map(frequent_score_dict)
        final_df['Total'] += final_df['Frequent_score']

        # 调整列的位置
        final_df = final_df[[col for col in final_df.columns if col != 'Total'] + ['Total']]

        return final_df, None  # Total_score 已经整合到 final_df

    def select_max_motif(self, final_df):
        """选择每个 Number 中分数最高的 motif"""
        max_motif = final_df.loc[final_df.groupby('Number')['Total'].idxmax()]
        return max_motif

    def plot_motif_trends(self, max_motif):
        """绘制 motif 变化趋势图并保存"""
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
        """选择不同 rank 中分数最高的 motif"""
        rank_threshold = max_motif.shape[0] / 20  # 5%
        valid_ranks = max_motif['Rank'].value_counts()
        valid_ranks = valid_ranks[valid_ranks > rank_threshold].index
        non_redundant_motif = max_motif[max_motif['Rank'].isin(valid_ranks)]
        non_redundant_motif = non_redundant_motif.loc[non_redundant_motif.groupby('Rank')['Total'].idxmax()]
        non_redundant_motif = non_redundant_motif.reset_index(drop=True)
        return non_redundant_motif

    def save_results(self, max_motif):
        """保存分析结果到 CSV 文件"""
        max_motif['Accession'] = self.accession
        max_motif.to_csv(self.output_file, index=False)
        print(f"保存结果到: {self.output_file}")

    def generate_meme_file(self, non_redundant_motif, extend_meme_10):
        """生成 MEME 文件"""
        non_redundant_motif = non_redundant_motif.rename(columns={'Number_x': 'Number'})
        non_redundant_motif['Number'] = non_redundant_motif['Number'].astype(str)
        extend_meme_10['Number'] = extend_meme_10['Number'].astype(str)

        # 提取需要的列
        non_redundant_motif_subset = non_redundant_motif[['Number', 'Layer']]
        extend_meme_10_subset = extend_meme_10[['Number', 'Layer', 'Strand', 'Start', 'End', 
                                                 'Start_extend', 'End_extend', 'p-value', 'e-value', 
                                                 'Sequence', 'Motif', 'Extend_sequence']]

        # 合并数据
        non_redundant_sequence = pd.merge(non_redundant_motif_subset, extend_meme_10_subset, on=['Number', 'Layer'], how='inner')

        # 生成 MEME 文件
        generate_all_pwm_and_write_to_meme(non_redundant_sequence, self.motif_dir)
        print(f"生成 MEME 文件到: {self.meme_file}")

    def run_fimo_analysis(self):
        """运行 FIMO 分析，包括转换 GenBank 文件和运行 FIMO"""
        self.convert_and_run_fimo(
            accession=self.accession,
            motif_file=self.meme_file,
            output_dir=self.fimo_output_dir
        )

    def convert_and_run_fimo(self, accession, motif_file, output_dir):
        """转换 GenBank 文件为 FASTA 并运行 FIMO"""
        gbk_file = f"/home/public_new/dowlond_phage/phage_gbk/{accession}.gbk"
        output_fasta = gbk_file.replace(".gbk", ".fasta")

        # 检查 GenBank 文件是否存在
        if not os.path.exists(gbk_file):
            print(f"错误: GenBank 文件不存在: {gbk_file}")
            return

        # 解析 GenBank 文件为 FASTA 格式
        self.parse_genbank_to_fasta(gbk_file, output_fasta)

        # 使用 FIMO 进行 motif 搜索，设置阈值为 1e-6
        fimo_success = self.run_fimo(motif_file, output_fasta, output_dir, threshold=1e-6)

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
        """将 GenBank 文件转换为 FASTA 格式"""
        if not os.path.exists(gbk_file):
            print(f"错误: GenBank 文件不存在: {gbk_file}")
            return

        with open(output_fasta, 'w') as fasta_file:
            for record in SeqIO.parse(gbk_file, "genbank"):
                fasta_file.write(f">{record.id}\n")
                fasta_file.write(str(record.seq) + "\n")
        print(f"已将 GenBank 文件转换为 FASTA 格式: {output_fasta}")

    def run_fimo(self, motif_file, genome_fasta, output_dir, threshold=1e-6):
        """运行 FIMO 并捕获输出"""
        if not os.path.exists(motif_file):
            print(f"错误: Motif 文件未找到: {motif_file}")
            return False
        if not os.path.exists(genome_fasta):
            print(f"错误: 基因组 FASTA 文件未找到: {genome_fasta}")
            return False

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 设置 FIMO 命令，移除 --text 参数，并添加 --thresh 参数
        fimo_cmd = [
            "/home/hanzequan/meme/bin/fimo",  # FIMO 的路径
            "--oc", output_dir,              # 输出目录
            "--thresh", str(threshold),      # 阈值
            motif_file,                       # motif 文件路径
            genome_fasta                      # 基因组 FASTA 文件路径
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
        """将 FIMO 的 TSV 输出文件转换为新的 TSV 文件"""
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
        """读取 FIMO 结果"""
        try:
            fimo_results = pd.read_csv(self.fimo_results_tsv, sep='\t', comment='#')
            print(f"成功读取 FIMO 结果: {self.fimo_results_tsv}")
            return fimo_results
        except FileNotFoundError:
            print(f"错误: FIMO 结果文件不存在: {self.fimo_results_tsv}")
            return None
