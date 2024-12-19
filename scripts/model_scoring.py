import numpy as np
import pandas as pd
from feature import *
def compute_average_probabilities(extend_meme_10, rf, numbers):
    """
    计算每个 Number 和 Layer 的平均概率。

    参数:
        extend_meme_10 (pd.DataFrame): 包含序列数据的数据框。
        rf (model): 已加载的随机森林模型。
        numbers (list): 要处理的 Number 列表。

    返回:
        pd.DataFrame: 包含 Number, Layer 和 Average_Probability 的结果数据框。
    """
    results_list = []
    for number in numbers:
        layers = range(1, 4)
        for layer in layers:
            layer_str = f"motif_{layer}"
            sequences = extend_meme_10[
                (extend_meme_10['Number'] == number) & 
                (extend_meme_10['Layer'] == layer_str)
            ]['Sequence'].to_list()
            if not sequences:
                continue
            results = predict_new_sequences(sequences, rf)
            right_column = results["probabilities"][:, 1]
            average_probability = np.mean(right_column)
            results_list.append([number, layer_str, average_probability])
    
    results_df = pd.DataFrame(results_list, columns=['Number', 'Layer', 'Average_Probability'])
    results_df['Number'] = results_df['Number'].astype(int)
    results_df = results_df.sort_values(by=['Number', 'Layer'])
    results_df['Number_Layer'] = results_df['Number'].astype(str) + "_" + results_df['Layer']
    
    return results_df

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


