import pandas as pd
import argparse
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
import re
from Bio.SeqRecord import SeqRecord
import time
#@title install and load model

import torch
from Bio import SeqIO
import random
import numpy as np
import matplotlib.pyplot as plt

# load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = "/home/hanzequan/megaDNA/megaDNA_phage_145M.pt" # model name
model = torch.load(model_path, map_location=torch.device(device))
model.eval()  # Set the model to evaluation mode

##算法##
import numpy as np
import torch
from collections import OrderedDict

# Alphabet 类，用于序列编码和解码
class Alphabet:
    def __init__(self):
        self.nt = ['**', 'A', 'T', 'C', 'G', '#']  # Alphabet
        self.ln = 4  # 核苷酸数量
        self.map_encode = OrderedDict((n, i) for i, n in enumerate(self.nt))
        self.map_decode = OrderedDict((v, k) for k, v in self.map_encode.items())
        self.start = [0]
        self.end = [5]
        self.codon = list(self.map_encode.values())[1:-1]

    def encode(self, sequence):
        # 将序列编码为数字
        x = [self.map_encode[s] if s in self.map_encode.keys() else 1 for s in sequence]
        return self.start + x + self.end

    def decode(self, x):
        # 将数字解码为序列
        return [self.map_decode[i] for i in x]

# 计算雅可比矩阵的函数
def get_categorical_jacobian(seq, model, device, alphabet):
    with torch.no_grad():
        x, ln = torch.tensor(alphabet.encode(seq)).unsqueeze(0).to(device), len(seq)
        f = lambda k: model(k, return_value='logits')[..., 1:-1, 1:5].cpu().numpy()
        
        # 计算原始的 f(x)
        fx = f(x)[0]
        fx_h = np.zeros((ln, alphabet.ln, ln, alphabet.ln))

        # 克隆多个 x 并逐一修改
        xs = torch.tile(x, [alphabet.ln, 1])
        for n in range(ln):
            x_h = torch.clone(xs)
            x_h[:, n+1] = torch.tensor(alphabet.codon)
            fx_h[n] = f(x_h)
        
        return fx_h - fx  # 返回 f(x+h) - f(x)

# 去均值或奇异值分解修正函数
def do_apc(x, rm=1):
    x = np.copy(x)
    if rm == 0:
        return x
    elif rm == 1:
        # 使用均值修正
        a1 = x.sum(0, keepdims=True)
        a2 = x.sum(1, keepdims=True)
        y = x - (a1 * a2) / x.sum()
    else:
        # 使用奇异值分解修正
        u, s, v = np.linalg.svd(x)
        y = s[rm:] * u[:, rm:] @ v[rm:, :]
    
    np.fill_diagonal(y, 0)  # 对角线置零
    return y

# 将雅可比矩阵转换为接触矩阵
def get_contacts(jacobian, symm=True, center=True, rm=1):
    j = jacobian.copy()
    
    # 中心化
    if center:
        for i in range(4):
            j -= j.mean(i, keepdims=True)
    
    # 计算 Frobenius 范数并修正
    j_fn = np.sqrt(np.square(j).sum((1, 3)))
    np.fill_diagonal(j_fn, 0)
    j_fn_corrected = do_apc(j_fn, rm=rm)
    
    # 对称化
    if symm:
        j_fn_corrected = (j_fn_corrected + j_fn_corrected.T) / 2
    
    return j_fn_corrected

# 将矩阵缩小为 N x N
def shrink_matrix(matrix, N):
    assert matrix.shape[0] == 3 * N and matrix.shape[1] == 3 * N, "Matrix must be of size 3N x 3N"
    shrunk_matrix = np.zeros((N, N))
    
    # 计算每个 3x3 块的均值
    for i in range(N):
        for j in range(N):
            shrunk_matrix[i, j] = np.mean(matrix[3*i:3*i+3, 3*j:3*j+3])
    
    return shrunk_matrix


###

def process_gbk_file(gbk_file, pfam_hmm, output_dir_base, protein_names, exclude_keywords, model, device):
    """
    处理单个 GenBank 文件，提取序列，计算雅可比矩阵和接触矩阵，保存结果。
    """
    # 获取 GenBank 文件的基本名称，用于目录命名
    gbk_basename = os.path.splitext(os.path.basename(gbk_file))[0]

    # 定义该 GenBank 文件的输出目录
    output_dir = os.path.join(output_dir_base, gbk_basename)

    # 检查输出目录是否存在，若存在则认为已处理，跳过处理
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Skipping processing of {gbk_file}.")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义临时目录（在基因组的输出目录下）
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # 定义输出的 FASTA 文件路径
    output_fasta = os.path.join(output_dir, 'extracted_sequences.fasta')

    # 提取序列，并保存 HMMScan 结果
    results = extract_protein_and_intergenic_sequences(
        gbk_file=gbk_file,
        temp_dir=temp_dir,
        protein_names=protein_names,
        pfam_hmm=pfam_hmm,
        exclude_keywords=exclude_keywords,
        output_dir=output_dir,  # 传递输出目录用于保存 HMMScan 结果
        output_fasta=output_fasta  # 传递输出的 FASTA 文件路径
    )

    if results is None:
        print(f"Failed to extract sequences from {gbk_file}, skipping.")
        return

    nucleotide_seq, protein_seq, intergenic_sequences, domain_nt_positions, strand, gene_start, gene_end, intergenic_sequences_info = results

    if not nucleotide_seq or not protein_seq:
        print(f"Failed to extract sequences from {gbk_file}, skipping.")
        return

    # 处理序列
    try:
        process_sequences(output_fasta, temp_dir, output_dir, domain_nt_positions, model, device)
    except Exception as e:
        print(f"Error during processing sequences for {gbk_file}: {e}")
        return

    # 清理临时文件（可选）
    try:
        os.remove(os.path.join(temp_dir, "protein_seq.fasta"))
        os.remove(os.path.join(temp_dir, "hmmscan_output.txt"))
        os.rmdir(temp_dir)  # 删除临时目录
    except OSError as e:
        print(f"Error removing temporary files: {e}")

def extract_protein_and_intergenic_sequences(gbk_file, temp_dir, protein_names, pfam_hmm, exclude_keywords=None, output_dir=None, output_fasta=None):
    from Bio.Seq import Seq
    import shutil

    # 设置默认的排除关键词
    if exclude_keywords is None:
        exclude_keywords = ['Cro', 'anti', 'cro', 'CRO']

    # 检查输入的GenBank文件是否存在
    if not os.path.exists(gbk_file):
        print(f"文件 {gbk_file} 不存在。")
        return None

    # 解析GenBank文件
    records = list(SeqIO.parse(gbk_file, "genbank"))
    if not records:
        print(f"无法解析GenBank文件 {gbk_file}。")
        return None

    record = records[0]  # 假设只有一个记录
    genome_seq = record.seq

    # 收集所有CDS特征
    cds_features = [feature for feature in record.features if feature.type == "CDS" and "product" in feature.qualifiers]

    if not cds_features:
        print("GenBank文件中没有CDS特征。")
        return None

    # 对CDS特征按起始位置排序
    cds_features_sorted = sorted(cds_features, key=lambda f: f.location.start)

    # 寻找第一个符合条件的蛋白质
    matching_cds = None
    for feature in cds_features_sorted:
        product = feature.qualifiers["product"][0]

        # 排除包含不需要的关键词的蛋白质（不区分大小写）
        if any(keyword.lower() in product.lower() for keyword in exclude_keywords):
            continue

        # 检查蛋白质名称是否在指定列表中（不区分大小写）
        if any(protein_name.lower() in product.lower() for protein_name in protein_names):
            matching_cds = feature
            break  # 找到第一个匹配的CDS，退出循环

    if not matching_cds:
        print("未找到符合条件的蛋白质。")
        return None

    # 获取基因位置信息
    location = matching_cds.location
    gene_start = int(location.start)  # Biopython使用0-based索引
    gene_end = int(location.end)
    strand = location.strand  # +1 或 -1

    # 提取匹配CDS的核苷酸序列
    try:
        # 提取核苷酸序列（保持原始方向）
        nucleotide_seq = matching_cds.extract(record.seq)

        # 获取翻译后的氨基酸序列
        protein_seq = matching_cds.qualifiers.get("translation", [""])[0]
        if not protein_seq:
            # 根据链方向翻译蛋白质序列
            if strand == 1:
                protein_seq = nucleotide_seq.translate(to_stop=True)
            elif strand == -1:
                protein_seq = nucleotide_seq.reverse_complement().translate(to_stop=True)
            else:
                print("未知的链方向。")
                return None
    except Exception as e:
        print(f"提取序列时出错: {e}")
        return None

    if not nucleotide_seq or not protein_seq:
        print("未能提取到蛋白质的核苷酸序列或氨基酸序列。")
        return None

    # 使用 SeqRecord 对象准备 FASTA 条目
    strand_symbol = '+' if strand == 1 else '-'

    # 对负链的序列进行反向互补操作，然后创建 SeqRecord
    fasta_sequence_protein_nucleotide = str(nucleotide_seq)
    if strand == -1:
        fasta_sequence_protein_nucleotide = str(Seq(fasta_sequence_protein_nucleotide).reverse_complement())

    # 创建蛋白质核苷酸序列的 SeqRecord
    protein_nt_record = SeqRecord(
        Seq(fasta_sequence_protein_nucleotide),
        id=f"{record.id}_{matching_cds.qualifiers['product'][0].replace(' ', '_')}_nucleotide",
        description=f"Location:{gene_start+1}-{gene_end}({strand_symbol})"
    )

    # 收集基因间序列及其信息
    intergenic_records = []
    intergenic_sequences_info = []

    # 修改部分：只提取基因之间的间隔区，不包括基因组开始到第一个基因，和最后一个基因到基因组结束的序列
    for i in range(len(cds_features_sorted) - 1):
        current_feature = cds_features_sorted[i]
        next_feature = cds_features_sorted[i + 1]
        current_end = int(current_feature.location.end)
        next_start = int(next_feature.location.start)

        if current_end < next_start:
            seq = genome_seq[current_end:next_start]
            intergenic_sequences_info.append({
                'genome_start': current_end + 1,  # 调整为1-based索引
                'genome_end': next_start
            })
            fasta_header = f"{record.id}_Intergenic_{current_end+1}_{next_start}"
            intergenic_record = SeqRecord(
                Seq(str(seq)),
                id=fasta_header,
                description=f"Location:{current_end+1}-{next_start}(+)"
            )
            intergenic_records.append(intergenic_record)

    # 将所有序列保存到 FASTA 文件
    records_to_write = [protein_nt_record] + intergenic_records
    SeqIO.write(records_to_write, output_fasta, "fasta")
    print(f"序列已保存到 {output_fasta}")

    # 运行hmmscan，寻找HTH结构域
    domain_nt_positions = find_helix_turn_helix_domain(
        nucleotide_seq=str(nucleotide_seq),
        protein_seq=str(protein_seq),
        pfam_hmm=pfam_hmm,
        temp_dir=temp_dir,
        strand=strand,
        output_dir=output_dir  # 保存 HMMScan 结果
    )

    if domain_nt_positions is None:
        print("未能找到HTH结构域，程序终止。")
        return None

    return str(nucleotide_seq), str(protein_seq), intergenic_records, domain_nt_positions, strand, gene_start, gene_end, intergenic_sequences_info

def find_helix_turn_helix_domain(nucleotide_seq, protein_seq, pfam_hmm, temp_dir, strand, output_dir):
    """
    使用hmmscan在蛋白质序列中寻找多个抑制蛋白相关的结构域，并根据基因方向计算其核苷酸位置。

    参数：
    - nucleotide_seq (str): 目标蛋白质的核苷酸序列（对于负链，已反向互补）。
    - protein_seq (str): 蛋白质的氨基酸序列。
    - pfam_hmm (str): Pfam HMM数据库的路径。
    - temp_dir (str): 用于存放临时文件的目录。
    - strand (int): 基因的链方向（+1 或 -1）。
    - output_dir (str): 输出目录，用于保存 HMMScan 结果。

    返回：
    - domain_nt_positions (tuple): HTH结构域在核苷酸序列中的起始和结束位置（已扩展20 bp）。
    """
    import subprocess
    from Bio.Seq import Seq
    import shutil

    # 定义目标结构域描述的集合
    target_domain_descriptions = [
        'Helix-turn-helix',
        'Bacterial regulatory proteins',
        'Repressor',
        'Transcriptional repressor',
        'HTH domain',
        'Regulatory protein',
        'DNA-binding domain',
        'Arc-like DNA binding domain',
        'Nucleopolyhedrovirus P10 protein'
    ]

    # 将蛋白质序列保存到临时FASTA文件
    protein_fasta = os.path.join(temp_dir, "protein_seq.fasta")
    with open(protein_fasta, "w") as f:
        f.write(f">protein_of_interest\n{protein_seq}\n")

    # 运行hmmscan
    hmmscan_output = os.path.join(temp_dir, "hmmscan_output.txt")
    command = [
        "hmmscan",
        "--domtblout", hmmscan_output,
        pfam_hmm,
        protein_fasta
    ]

    try:
        print("正在运行hmmscan...")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("hmmscan运行完成。")
    except subprocess.CalledProcessError as e:
        print(f"运行hmmscan时出错: {e}")
        return None

    # 将 hmmscan_output.txt 复制到输出目录
    output_hmmscan = os.path.join(output_dir, "hmmscan_output.txt")
    shutil.copyfile(hmmscan_output, output_hmmscan)
    print(f"HMMScan 结果已保存到 {output_hmmscan}")

    # 解析hmmscan输出，寻找目标结构域
    target_domains = []
    with open(hmmscan_output, "r") as infile:
        for line in infile:
            if line.startswith("#"):
                continue
            columns = line.strip().split()
            if len(columns) < 23:
                continue
            description = ' '.join(columns[22:])
            # 检查描述中是否包含任何目标结构域描述
            if any(target_desc.lower() in description.lower() for target_desc in target_domain_descriptions):
                try:
                    domain_score = float(columns[13])
                    ali_from = int(columns[17])
                    ali_to = int(columns[18])
                    target_domains.append({
                        'description': description,
                        'domain_score': domain_score,
                        'ali_from': ali_from,
                        'ali_to': ali_to
                    })
                except (ValueError, IndexError) as parse_error:
                    print(f"解析行时出错: {parse_error}")
                    continue

    if not target_domains:
        print("在hmmscan结果中未找到目标结构域。将整个序列视为一个结构域。")
        best_domain = {
            'description': 'Entire_sequence',
            'domain_score': 0.0,  # 可以根据需要设置默认得分
            'ali_from': 1,  # 通常蛋白质序列从1开始
            'ali_to': len(protein_seq)
        }
    else:
        # 选择得分最高的结构域
        best_domain = max(target_domains, key=lambda x: x['domain_score'])
        print(f"找到的最佳结构域: {best_domain['description']}, 得分: {best_domain['domain_score']}")
        print(f"氨基酸位置: {best_domain['ali_from']}-{best_domain['ali_to']}")

    gene_length = len(nucleotide_seq)

    # 计算结构域在核苷酸序列中的位置
    if strand == 1:
        # 正链，从前向后
        nt_start = (best_domain['ali_from'] - 1) * 3
        nt_end = best_domain['ali_to'] * 3 - 1
    elif strand == -1:
        # 负链，从后向前，需要反向计算
        nt_end = gene_length - (best_domain['ali_from'] - 1) * 3 - 1
        nt_start = gene_length - best_domain['ali_to'] * 3
    else:
        print("未知的链方向。")
        return None

    # 扩展20 bp，并确保不越界
    domain_nt_start_in_seq = max(0, nt_start - 20)
    domain_nt_end_in_seq = min(gene_length - 1, nt_end + 20)

    print(f"结构域在核苷酸序列中的位置（扩展20 bp）: {domain_nt_start_in_seq}-{domain_nt_end_in_seq}")

    return (domain_nt_start_in_seq, domain_nt_end_in_seq)

def process_sequences(output_fasta, temp_dir, output_dir, domain_nt_positions, model, device):
    """
    从 output_fasta 中读取序列，处理它们，计算雅可比矩阵和接触矩阵，保存结果。
    """
    # 读取 output_fasta 中的序列
    sequences = []
    sequence_headers = []
    sequence_sources = []
    sequence_genomic_info = {}

    location_pattern = re.compile(r'Location:(\d+)-(\d+)\((\+|-)\)')

    for idx, record in enumerate(SeqIO.parse(output_fasta, 'fasta')):
        sequences.append(str(record.seq))
        sequence_headers.append(record.id)
        if 'Intergenic' in record.id:
            sequence_sources.append('intergenic')
        else:
            sequence_sources.append('protein')

        # 从 record.description 中解析 Location 信息
        match = location_pattern.search(record.description)
        if match:
            start, end, strand = match.groups()
            sequence_genomic_info[idx] = {
                'genomic_start': int(start),
                'genomic_end': int(end),
                'strand': strand,
                'sequence': str(record.seq)
            }
        else:
            sequence_genomic_info[idx] = {
                'genomic_start': None,
                'genomic_end': None,
                'strand': None,
                'sequence': str(record.seq)
            }

    # 处理序列
    process_rounds(sequences, sequence_headers, sequence_genomic_info, output_dir, domain_nt_positions, model, device)

def process_rounds(sequences, sequence_headers, sequence_genomic_info, output_dir, domain_nt_positions, model, device):
    """
    以轮次处理序列，计算雅可比矩阵和接触矩阵，保存结果。
    """
    alphabet = Alphabet()

    # 定义用于保存更新后序列和结果的目录
    output_subdir = os.path.join(output_dir, 'updated_sequences')
    os.makedirs(output_subdir, exist_ok=True)

    # 检查是否至少有一个序列
    if not sequences:
        print(f"No sequences found to process.")
        return
    else:
        # 获取第一个序列（蛋白质序列）
        first_sequence = sequences[0]
        first_sequence_length = len(first_sequence)

        used_indices = set()
        round_count = 1
        sequence_indices = list(range(1, len(sequences)))

        while True:
            print(f"\nStarting round {round_count}...")

            final_sequence = first_sequence
            total_length = len(final_sequence)
            used_indices_in_round = []
            sequence_positions = {}
            sequence_positions[0] = (0, len(first_sequence) - 1)

            # 计算剩余长度
            remaining_length = sum(len(sequences[idx]) for idx in sequence_indices if idx not in used_indices)

            if remaining_length < first_sequence_length / 2:
                print("Remaining sequence length less than half of protein sequence, stopping.")
                break
            TARGET_LENGTH = 1500
            MAX_LENGTH = 1800

            for idx in sequence_indices:
                if idx in used_indices:
                    continue
                next_sequence = sequences[idx]
                next_length = len(next_sequence)

                if total_length + next_length <= TARGET_LENGTH:
                    final_sequence += next_sequence
                    start_pos = total_length
                    end_pos = total_length + next_length - 1
                    sequence_positions[idx] = (start_pos, end_pos)
                    total_length += next_length
                    used_indices_in_round.append(idx)
                    used_indices.add(idx)
                elif total_length + next_length <= MAX_LENGTH:
                    final_sequence += next_sequence
                    start_pos = total_length
                    end_pos = total_length + next_length - 1
                    sequence_positions[idx] = (start_pos, end_pos)
                    total_length += next_length
                    used_indices_in_round.append(idx)
                    used_indices.add(idx)
                    break
                else:
                    continue

            if not used_indices_in_round:
                print("No unused sequences can be added, stopping.")
                break

            print(f"Combined sequence length is {total_length} bp, contains sequence indices: [0] + {used_indices_in_round}")

            # 更新序列标题
            updated_headers = []
            for idx in [0] + used_indices_in_round:
                header = sequence_headers[idx]
                start_pos, end_pos = sequence_positions[idx]
                updated_header = f"{header}|Relative_Position:{start_pos}-{end_pos}"
                updated_headers.append(updated_header)

            # 写入更新后的序列到新的 FASTA 文件
            seq_records = []
            for idx, header in zip([0] + used_indices_in_round, updated_headers):
                seq = sequences[idx]
                record = SeqRecord(Seq(seq), id=header, description="")
                seq_records.append(record)

            updated_fasta_filename = f'updated_extracted_sequences_round_{round_count}.fasta'
            updated_fasta_path = os.path.join(output_subdir, updated_fasta_filename)
            SeqIO.write(seq_records, updated_fasta_path, 'fasta')
            print(f"Updated sequences written to {updated_fasta_path}")

            # 执行计算
            try:
                perform_calculations(final_sequence, sequence_positions, sequence_genomic_info, domain_nt_positions, round_count, output_dir, model, device)
            except Exception as e:
                print(f"Error during calculations in round {round_count}: {e}")
                # 继续处理下一轮
                round_count += 1
                continue

            # 检查是否所有序列都已使用
            if len(used_indices) == len(sequences) - 1:
                print("All sequences have been processed.")
                break

            round_count += 1

def perform_calculations(final_sequence, sequence_positions, sequence_genomic_info, domain_nt_positions, round_count, output_dir, model, device):
    """
    计算雅可比矩阵和接触矩阵，绘制并保存结果。
    """
    alphabet = Alphabet()
    sequence = final_sequence
    print('Sequence:', sequence)

    # 记录开始时间
    start_time = time.time()
    print("Starting Jacobian and contacts calculations...")

    # 计算雅可比矩阵
    jac = get_categorical_jacobian(sequence, model, device, alphabet)

    # 计算雅可比矩阵的平均值
    jac_mean = np.mean(jac)

    # 更新雅可比矩阵的指定值
    for i in range(len(sequence)):
        for j in range(4):
            sequence2 = list(sequence)
            sequence2[i] = alphabet.nt[1 + j]
            sequence2 = ''.join(sequence2)
            codon_start = (i // 3) * 3
            codon_end = codon_start + 3
            codon = sequence2[codon_start:codon_end]

            if codon in ['TTA', 'CTA', 'TCA']:
                jac[i, j, :, :] = jac_mean
                jac[:, :, i, j] = jac_mean

    # 对雅可比矩阵中心化并对称化
    for i in range(4):
        jac -= jac.mean(i, keepdims=True)
    jac = (jac + jac.transpose(2, 3, 0, 1)) / 2

    # 计算接触矩阵
    contacts = get_contacts(jac)

    # 绘制并保存图像
    contacts2 = contacts.copy()

    # 将下三角的元素设为零
    lower_triangle_indices = np.tril_indices(len(contacts), -1)
    contacts2[lower_triangle_indices] = 0

    # 创建图像
    plt.figure(figsize=(15, 15))
    plt.imshow(contacts2, cmap='Greys')
    plt.colorbar()
    plt.clim([0, 0.01])

    # 设置刻度
    plt.xticks(range(0, len(contacts), 100), rotation=90)
    plt.yticks(range(0, len(contacts), 100))

    # 添加蛋白质序列区域的线条
    protein_nt_start_in_seq = sequence_positions[0][0]
    protein_nt_end_in_seq = sequence_positions[0][1]
    plt.axhline(y=protein_nt_start_in_seq, color='green', linewidth=1, label='Protein Region Start')
    plt.axhline(y=protein_nt_end_in_seq, color='green', linewidth=1, label='Protein Region End')
    plt.axvline(x=protein_nt_start_in_seq, color='green', linewidth=1)
    plt.axvline(x=protein_nt_end_in_seq, color='green', linewidth=1)

    # 添加 HTH 结构域的线条
    domain_nt_start_in_seq, domain_nt_end_in_seq = domain_nt_positions
    hth_start_global = protein_nt_start_in_seq + domain_nt_start_in_seq
    hth_end_global = protein_nt_start_in_seq + domain_nt_end_in_seq
    plt.axhline(y=hth_start_global, color='blue', linewidth=1, label='HTH Domain Start')
    plt.axhline(y=hth_end_global, color='blue', linewidth=1, label='HTH Domain End')

    # 计算 HTH 互作
    sequence_length = len(sequence)
    protein_indices = set(range(protein_nt_start_in_seq, protein_nt_end_in_seq + 1))
    hth_indices = range(hth_start_global, hth_end_global + 1)

    # 初始化列表存储互作信息
    hth_interactions = []

    for i in hth_indices:
        for j in range(sequence_length):
            if j not in protein_indices:
                interaction_strength = contacts[i, j]
                if interaction_strength > 0.05:
                    hth_interactions.append({
                        'hth_position': i,
                        'other_position': j,
                        'interaction_strength': interaction_strength
                    })

    # 保存互作信息到 DataFrame
    hth_df = pd.DataFrame(hth_interactions)

    # 获取 unique 的 other_position
    unique_other_positions = hth_df['other_position'].unique()

    # 初始化列表，存储每个 other_position 的信息
    other_positions_info = []

    for pos in unique_other_positions:
        seq_idx = None
        for idx, (start, end) in sequence_positions.items():
            if start <= pos <= end:
                seq_idx = idx
                rel_pos_in_seq = pos - start
                break
        if seq_idx is None:
            continue

        seq_info = sequence_genomic_info.get(seq_idx, {})
        seq_seq = seq_info.get('sequence', '')

        if rel_pos_in_seq - 10 < 0 or rel_pos_in_seq + 10 >= len(seq_seq):
            continue

        sequence_context = seq_seq[rel_pos_in_seq - 10: rel_pos_in_seq + 11]

        genomic_start = seq_info.get('genomic_start')
        genomic_end = seq_info.get('genomic_end')
        strand = seq_info.get('strand')

        if genomic_start is None:
            absolute_position = None
        else:
            if strand == '+':
                absolute_position = genomic_start + rel_pos_in_seq - 1  # 调整为1-based索引
            elif strand == '-':
                absolute_position = genomic_end - rel_pos_in_seq + 1  # 调整为1-based索引
            else:
                absolute_position = None

        other_positions_info.append({
            'other_position': pos,
            'sequence_context': sequence_context,
            'absolute_genomic_position': absolute_position
        })

    other_positions_df = pd.DataFrame(other_positions_info)

    hth_df = hth_df.merge(other_positions_df, on='other_position', how='left')
    hth_df = hth_df.dropna(subset=['sequence_context'])

    # 保存 DataFrame 到 CSV
    interactions_csv_filename = f'hth_interactions_round_{round_count}.csv'
    interactions_csv_path = os.path.join(output_dir, interactions_csv_filename)
    hth_df.to_csv(interactions_csv_path, index=False)
    print(f"hth_interactions saved to {interactions_csv_path}")

    # 在图像上叠加 HTH 互作位置
    x_positions = hth_df['other_position'].tolist()
    y_positions = hth_df['hth_position'].tolist()
    interaction_strengths = hth_df['interaction_strength'].tolist()

    # 调整点的大小
    sizes = [max(1, (s - 0.05) * 100) for s in interaction_strengths]

    plt.scatter(x_positions, y_positions, color='red', s=sizes, label='HTH Interactions')

    # 添加图例
    plt.legend(loc='upper right')

    # 保存图像
    figure_filename = f'contacts_round_{round_count}_with_hth_interactions.png'
    figure_path = os.path.join(output_dir, figure_filename)
    plt.savefig(figure_path)
    plt.close()
    print(f"Figure with HTH interactions saved to {figure_path}")

    # 输出耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Round {round_count} completed, time elapsed: {elapsed_time:.2f} seconds")

def gene_interaction_main(input_dir, output_dir, pfam_hmm, protein_names, exclude_keywords, model, device):
    """
    主函数，用于处理输入目录中的所有 GenBank 文件。
    """
    # 创建基础目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义批处理输出目录
    output_dir_base = os.path.join(output_dir, 'batch_figure')
    os.makedirs(output_dir_base, exist_ok=True)
    
    # 获取所有 GenBank 文件
    gbk_files = [f for f in os.listdir(input_dir) if f.endswith('.gbk')]
    
    if not gbk_files:
        print(f"No GenBank files found in directory: {input_dir}")
        return
    
    for gbk_file in gbk_files:
        gbk_path = os.path.join(input_dir, gbk_file)
        print(f"Processing {gbk_path}...")
        try:
            process_gbk_file(gbk_path, pfam_hmm, output_dir_base, protein_names, exclude_keywords, model, device)
        except Exception as e:
            print(f"Error processing {gbk_path}: {e}")
            continue
if __name__ == "__main__":
    main()
