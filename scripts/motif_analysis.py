import os
import subprocess
import csv
import pandas as pd
import shutil
from Bio import SeqIO, motifs
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import numpy as np

# 导入自定义模块的函数
from feature import com_seq_feature
from write_to_file import *
from MPI_DFI import *
from analysis_meme_file import *
from model_scoring import *

class MotifAnalyzer:
    def __init__(self, accession, base_output_dir='output'):
        """
        初始化 MotifAnalyzer 类。

        :param accession: GenBank 文件对应的 accession ID
        :param base_output_dir: 基础输出目录，默认为当前目录下的 'output'
        """
        self.accession = accession
        self.base_output_dir = base_output_dir
        self.accession_dir = os.path.join(self.base_output_dir, accession)
        self.predict_path = os.path.join('exclude_GPD_find_key_motif', f'Branch_1_{accession}')
        self.genbank_dir = os.path.join('download_phage', 'phage_gbk')
        self.motif_dir = os.path.join(self.accession_dir, 'motif_meme')
        self.fimo_output_dir = os.path.join(self.motif_dir, 'fimo_results')
        self.output_file = os.path.join(self.accession_dir, f"{accession}.csv")
        self.meme_file = os.path.join(self.motif_dir, 'final_motif.meme')
        self.fimo_results_tsv = os.path.join(self.fimo_output_dir, 'fimo_results.tsv')
        self.extend_meme_10_path = os.path.join(self.accession_dir, 'extend_meme_10.csv')
        self.results_df_path = os.path.join(self.accession_dir, 'results_df.csv')
        self.rank_list_path = os.path.join(self.accession_dir, 'rank_list.csv')
        self.tomtom_results_path = os.path.join(self.fimo_output_dir, 'tomtom_results.tsv')
        self.non_redundant_meme_path = os.path.join(self.motif_dir, 'non_redundant.meme')

        # 创建必要的目录结构
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录结构"""
        os.makedirs(self.accession_dir, exist_ok=True)
        os.makedirs(self.motif_dir, exist_ok=True)
        os.makedirs(self.fimo_output_dir, exist_ok=True)
        print(f"创建目录: {self.accession_dir}")

    def run_analysis(self, rf):
        """
        运行完整的分析流程。

        :param rf: 预训练的随机森林模型
        :return: max_motif, non_redundant_motif, fimo_results
        """
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
        """构建 motif 矩阵并进行过滤，如果存在则加载"""
        if os.path.exists(self.extend_meme_10_path):
            extend_meme_10 = pd.read_csv(self.extend_meme_10_path)
            print(f"加载已有的 extend_meme_10: {self.extend_meme_10_path}")
        else:
            extend_meme_10 = build_motif_matrices(self.predict_path)
            # 如果需要对 extend_meme_10 进行过滤，可以在这里取消注释并调整
            # extend_meme_10 = extend_meme_10[
            #     (extend_meme_10['Number'].astype(int) >= 1) &
            #     (extend_meme_10['Number'].astype(int) <= 101)
            # ]
            extend_meme_10.to_csv(self.extend_meme_10_path, index=False)
            print(f"保存 extend_meme_10 到: {self.extend_meme_10_path}")
        return extend_meme_10

    def remove_redundant_sequences(self, extend_meme_10):
        """
        移除冗余序列并保存 rank_list。

        :param extend_meme_10: DataFrame，包含 motif 信息
        :return: rank_list DataFrame
        """
        if os.path.exists(self.rank_list_path):
            rank_list = pd.read_csv(self.rank_list_path)
            print(f"加载已有的 rank_list: {self.rank_list_path}")
        else:
            rank_list = remove_redundant_sequences(
                extend_meme_10,
                output_dir=self.motif_dir,  # 使用相对路径
                result_path=os.path.join(self.fimo_output_dir, 'tomtom_results'),
                final_output=self.non_redundant_meme_path
            )
            rank_list.to_csv(self.rank_list_path, index=False)
            print(f"保存 rank_list 到: {self.rank_list_path}")
        return rank_list

    def compute_average_probabilities(self, extend_meme_10, rf):
        """
        计算机器学习评分并保存 results_df。

        :param extend_meme_10: DataFrame，包含 motif 信息
        :param rf: 预训练的随机森林模型
        :return: results_df DataFrame
        """
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
        """
        计算总分和频率分数。

        :param merged_df: DataFrame，合并后的评分数据
        :param rank_list: DataFrame，rank 信息
        :return: final_df DataFrame
        """
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
        """
        运行 FIMO 分析，包括转换 GenBank 文件和运行 FIMO。
        """
        self.convert_and_run_fimo(
            accession=self.accession,
            motif_file=self.meme_file,
            output_dir=self.fimo_output_dir
        )

    def convert_and_run_fimo(self, accession, motif_file, output_dir):
        """
        转换 GenBank 文件为 FASTA 并运行 FIMO。

        :param accession: GenBank 文件对应的 accession ID
        :param motif_file: motif 文件路径（MEME 格式）
        :param output_dir: FIMO 搜索结果的输出目录
        """
        gbk_file = os.path.join(self.genbank_dir, f'{accession}.gbk')
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

    def run_fimo(self, motif_file, genome_fasta, output_dir, threshold=1e-6):
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

        # 设置 FIMO 命令，使用相对路径
        fimo_cmd = [
            os.path.join('meme', 'bin', 'fimo'),  # FIMO 的相对路径
            "--oc", output_dir,                   # 输出目录
            "--thresh", str(threshold),           # 阈值
            motif_file,                           # motif 文件路径
            genome_fasta                          # 基因组 FASTA 文件路径
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
            fimo_results = pd.read_csv(self.fimo_results_tsv, sep='\t', comment='#')
            print(f"成功读取 FIMO 结果: {self.fimo_results_tsv}")
            return fimo_results
        except FileNotFoundError:
            print(f"错误: FIMO 结果文件不存在: {self.fimo_results_tsv}")
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

            write_motif_to_file(pwm_df, meme_output_path, layer, number, append=True)

    print(f"生成 MEME 文件到: {meme_output_path}")

def write_motif_to_file(pwm_df, file_path, motif_name, number, append=True):
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

        file.write(f'MOTIF {number}_{motif_name}\n')
        file.write(f'letter-probability matrix: alength= 4 w= {w} nsites= 20 E= 0\n')

        pwm_string = pwm_df.to_string(index=False, header=False)
        file.write(pwm_string + '\n\n')

    print(f"写入 motif {number}_{motif_name} 到 MEME 文件")

def run_tomtom_small_e_value(result_path, query_motif, target_motif):
    """
    运行 Tomtom 分析，并返回结果。

    :param result_path: Tomtom 结果输出目录
    :param query_motif: 查询 motif 文件路径
    :param target_motif: 目标 motif 文件路径
    :return: Tomtom 结果 DataFrame 或 None
    """
    tomtom_command = [
        os.path.join('meme', 'bin', 'tomtom'),
        "-dist", "ed",            # 使用欧氏距离作为相似性度量，更严格
        "-min-overlap", "8",      # 要求至少8个碱基位点的重叠
        "-thresh", "1e-10",       # 更低的阈值，更严格的筛选
        "-oc", result_path,
        query_motif,
        target_motif,
        '-verbosity', '1'
    ]

    try:
        subprocess.run(tomtom_command, check=True)
        print("Tomtom 分析成功完成。")
    except subprocess.CalledProcessError as e:
        print(f"运行 Tomtom 时发生错误: {e}")
        return None

    output_file = os.path.join(result_path, 'tomtom.tsv')
    return self.read_tomtom_results(output_file)

def read_tomtom_results(output_file):
    """
    读取 Tomtom 分析的结果。

    :param output_file: Tomtom 结果文件路径
    :return: Tomtom 结果 DataFrame 或 None
    """
    try:
        tomtom_results = pd.read_csv(output_file, sep='\t')
        print(f"成功读取 Tomtom 结果: {output_file}")
        return tomtom_results
    except FileNotFoundError:
        print(f"错误: Tomtom 结果文件不存在: {output_file}")
        return None

def write_non_redundant_meme(tfbs_df, keep_motifs, final_output):
    """
    写出非冗余的 MEME 文件。

    :param tfbs_df: DataFrame，包含 motif 信息
    :param keep_motifs: 列表，需保留的 motif ID
    :param final_output: 输出的 MEME 文件路径
    """
    non_redundant_df = tfbs_df[tfbs_df['Motif_ID'].isin(keep_motifs)]
    generate_all_pwm_and_write_to_meme(non_redundant_df, os.path.dirname(final_output))
    print(f"写出非冗余的 MEME 文件到: {final_output}")

# 示例使用
if __name__ == "__main__":
    # 加载或定义你的随机森林模型
    # 例如:
    # import pickle
    # with open('path_to_rf_model.pkl', 'rb') as f:
    #     rf = pickle.load(f)
    rf_model_path = '/home/hanzequan/test_bectiral/rf_model/tfbs_model.joblib'
    # 加载随机森林模型
    rf = load(rf_model_path)

    # 定义 accession ID
    accession = 'NC_005856'

    # 创建 MotifAnalyzer 实例
    analyzer = MotifAnalyzer(accession=accession)

    # 运行分析
    max_motif, non_redundant_motif, fimo_results = analyzer.run_analysis(rf)

    # 你可以在这里根据需要进一步处理 fimo_results
    if fimo_results is not None:
        print(fimo_results.head())

