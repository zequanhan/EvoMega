import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN

def calculate_ratio(extend_meme_10):
    print('基础距离函数')
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
            # 基于参与近邻关系的序列比例（参与度评分）
            # 定义：让有至少一个近邻序列（即参与过close_sequences的序列）的数量作为分子，总序列数作为分母。
            close_sequences_set = set()
            for pair in close_sequences:
                close_sequences_set.add(pair['Sequence_1'])
                close_sequences_set.add(pair['Sequence_2'])

            ratio = len(close_sequences_set) / len(filtered_data)
            # 存储结果
            results.append({
                'Number': int(number),
                'Number_Layer': f'{number}_{layer}',
                'Ratio': ratio
            })

    # 将结果转换为 DataFrame
    distance_result_df = pd.DataFrame(results)
    return distance_result_df
def calculate_total_scores(merged_df, rank_list):
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
