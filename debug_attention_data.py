"""
debug_attention_data.py
检查保存的注意力数据
"""

import numpy as np
import pickle


def debug_data():
    print("=" * 60)
    print("调试注意力数据")
    print("=" * 60)

    # 1. 检查统计量文件
    print("\n1. 检查统计量文件:")
    mean_attn = np.load('./analyze_attention/attn_stats_mean.npy')
    print(f"   mean_attn shape: {mean_attn.shape}")
    print(f"   mean_attn dtype: {mean_attn.dtype}")
    print(f"   有效值数量: {np.sum(~np.isnan(mean_attn))}")
    print(f"   NaN数量: {np.sum(np.isnan(mean_attn))}")
    print(f"   最小值: {np.nanmin(mean_attn)}")
    print(f"   最大值: {np.nanmax(mean_attn)}")

    # 检查第一个有效网格
    valid_idx = np.where(~np.isnan(mean_attn).all(axis=(2, 3, 4)))
    if len(valid_idx[0]) > 0:
        i, j = valid_idx[0][0], valid_idx[1][0]
        print(f"\n   第一个有效网格点: ({i}, {j})")
        print(f"   mean_attn[{i},{j}]:")
        for head in range(4):
            print(f"     Head {head}:")
            print(mean_attn[i, j, head])

    # 2. 检查完整分布文件
    print("\n2. 检查完整分布文件:")
    with open('./analyze_attention/attn_full_distributions.pkl', 'rb') as f:
        distributions = pickle.load(f)

    print(f"   网格点数量: {len(distributions)}")

    # 检查前几个网格点的数据
    for idx, (pos, data) in enumerate(list(distributions.items())[:5]):
        print(f"\n   网格点 {idx + 1}: {pos}")
        print(f"     attn shape: {data['attn'].shape}")
        print(f"     attn dtype: {data['attn'].dtype}")
        print(f"     attn 是否有NaN: {np.isnan(data['attn']).any()}")
        print(f"     attn 范围: [{data['attn'].min():.4f}, {data['attn'].max():.4f}]")
        print(f"     attn 均值: {data['attn'].mean():.4f}")
        print(f"     errors shape: {data['errors'].shape}")
        print(f"     n_samples: {data['n_samples']}")

    # 3. 检查原始的注意力权重是否合理
    print("\n3. 检查原始注意力权重合理性:")
    # 注意力权重应该是归一化的，每行和为1
    for idx, (pos, data) in enumerate(list(distributions.items())[:3]):
        attn = data['attn']  # (n_samples, heads, 3, 3)
        print(f"\n   网格点 {pos}:")
        # 检查第一行（ERA5作为查询）
        for head in range(min(2, attn.shape[1])):
            row_sum = attn[0, head, 0, :].sum()
            print(f"     Head {head}, 第一行和: {row_sum:.4f}")
            if abs(row_sum - 1.0) > 0.01:
                print(f"     警告：注意力权重未归一化！")


if __name__ == "__main__":
    debug_data()