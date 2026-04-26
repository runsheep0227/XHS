import pandas as pd
import numpy as np
import json
import math

# ---- 手动实现 chi2_contingency ----
def chi2_test(obs):
    """手动实现卡方独立性检验，返回 chi2, p, dof, expected"""
    obs = np.array(obs, dtype=float)
    n_rows, n_cols = obs.shape
    dof = (n_rows - 1) * (n_cols - 1)

    # 期望频数
    row_totals = obs.sum(axis=1, keepdims=True)
    col_totals = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    expected = row_totals * col_totals / total

    # χ² 统计量
    chi2 = np.sum((obs - expected) ** 2 / expected)

    # p值：Wilson-Hilferty 正态近似（df 大时足够精确）
    # 精确 p 值用 regularized incomplete gamma 级数近似
    from math import gamma as gamma_fn, exp, sqrt, pi, erf

    def chi2_cdf_approx(x, df):
        if x <= 0:
            return 0.0
        k = df / 2.0
        # 用 regularized gamma P(k, x/2) 的 series 展开近似（前 200 项）
        z = x / 2.0
        if z < k + 1:
            # series form
            s = 1.0 / k
            term = s
            for n in range(1, 200):
                term *= z / (k + n)
                s += term
                if abs(term) < 1e-12 * abs(s):
                    break
            result = s * exp(-z) * (z ** k) / gamma_fn(k)
        else:
            # continued fraction form，反转求 complement
            # 用 1 - F(x) ≈ Q(k, z) 的 series approximation
            # 直接用 Wilson-Hilferty 正态近似（df>=5 时精度 < 0.001）
            if df >= 5:
                t = (z / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))
                s = sqrt(9.0 * k)
                z_norm = t * s
                result = 0.5 * (1.0 + erf(z_norm / sqrt(2.0)))
            else:
                # 直接用 series（z < k+1 的情况）
                s = 1.0 / k
                term = s
                for n in range(1, 200):
                    term *= z / (k + n)
                    s += term
                    if abs(term) < 1e-12 * abs(s):
                        break
                result = s * exp(-z) * (z ** k) / gamma_fn(k)
        return min(1.0, max(0.0, result))

    p_value = 1.0 - chi2_cdf_approx(chi2, dof)
    return chi2, p_value, dof, expected

print('scipy 未安装，已使用纯 numpy 实现 chi2 检验\n')

# 1. 读取主题数据
topic_df = pd.read_csv(
    'content/bertopic_results_optimized/final_pro_topics.csv',
    usecols=['note_id', 'macro_topic_name', 'is_noise']
)
topic_df = topic_df[topic_df['is_noise'] == False].copy()
print(f'主题数据：{len(topic_df)} 条（去噪声后）')

mt_col = 'macro_topic_name'
print('\n宏观主题分布：')
print(topic_df[mt_col].value_counts())

# 2. 读取情感数据
with open('comment/comment_results/predicted_comments.json', 'r', encoding='utf-8') as f:
    comments = json.load(f)
sent_df = pd.DataFrame(comments)
print(f'\n情感数据：{len(sent_df)} 条')

# 3. 合并
merged = pd.merge(topic_df, sent_df, on='note_id', how='inner')
print(f'合并后：{len(merged)} 条')

# 4. 列联表
contingency = pd.crosstab(merged[mt_col], merged['sentiment_polarity'])
contingency.columns = ['负向(-1)', '中性(0)', '正向(+1)']
print('\n=== 列联表（宏观主题 × 情感极性）===')
print(contingency)

# 5. 卡方检验
chi2, p, dof, expected = chi2_test(
    contingency[['负向(-1)', '中性(0)', '正向(+1)']].values
)
print(f'\n=== 卡方检验结果 ===')
print(f'卡方统计量 χ² = {chi2:.4f}')
print(f'自由度 df = {dof}')
print(f'p值 = {p:.2e}')
sig = '拒绝原假设（主题类别与情感极性存在统计学显著关联）' if p < 0.05 else '不拒绝原假设'
print(f'α = 0.05 → {sig}')

# 6. Cramér V
n = contingency[['负向(-1)', '中性(0)', '正向(+1)']].sum().sum()
min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f'\nCramér V = {cramers_v:.4f}')
if cramers_v < 0.1:
    effect = '极弱关联'
elif cramers_v < 0.3:
    effect = '弱关联'
elif cramers_v < 0.5:
    effect = '中等关联'
else:
    effect = '强关联'
print(f'效应量：{effect}')

# 7. 行百分比
print('\n=== 各主题情感分布（行百分比，%）===')
row_pct = contingency[['负向(-1)', '中性(0)', '正向(+1)']].div(
    contingency[['负向(-1)', '中性(0)', '正向(+1)']].sum(axis=1), axis=0
) * 100
print(row_pct.round(2).to_string())

# 8. 标准化残差
obs = contingency[['负向(-1)', '中性(0)', '正向(+1)']].values
std_resid = (obs - expected) / np.sqrt(expected)
resid_df = pd.DataFrame(
    std_resid,
    index=contingency.index,
    columns=['负向残差', '中性残差', '正向残差']
)
print('\n=== 标准化残差（|残差|>2 提示显著正/负偏离期望值）===')
print(resid_df.round(3).to_string())

# 9. 各主题样本量
print('\n各主题样本量：')
totals = contingency[['负向(-1)', '中性(0)', '正向(+1)']].sum(axis=1).sort_values(ascending=False)
print(totals)
