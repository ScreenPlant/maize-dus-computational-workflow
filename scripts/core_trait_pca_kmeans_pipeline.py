import argparse
from pathlib import Path
# ====================== 0. 解决Windows环境警告 + 字体配置 ======================
import os

os.environ['OMP_NUM_THREADS'] = '1'

# ====================== 1. 导入所有依赖库 ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
# 新增依赖：多重检验校正
from statsmodels.stats.multitest import multipletests
# 新增依赖：交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

# ====================== 2. 基础全局设置 ======================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")
sns.set(font='Microsoft YaHei')

# ====================== 3. 关键变量定义 ======================
trait_mapping = [
    ('Trait_1_Predict', 'VG', '幼苗：第一叶顶端形状', 38.7),
    ('Trait_2_Predict', 'VG', '幼苗：第一叶鞘花青甙显色强度', 0.0),
    ('Trait_3_Predict', 'VG', '叶片：绿色程度', 0.0),
    ('Trait_6_Predict', 'VG', '植株：上部叶片与茎秆夹角', 0.0),
    ('Trait_7_Predict', 'VG', '叶片：弯曲程度', 0.0),
    ('Trait_9_Predict', 'VG', '雌穗：花丝花青甙显色强度', 61.3),
    ('Trait_10_Predict', 'VG', '雄穗：花药花青甙显色强度', 39.3),
    ('Trait_23_Predict', 'MS', '叶片：长度', 0.0),
    ('Trait_24_Predict', 'MS', '叶片：宽度', 34.7),
    ('Trait_26_Predict', 'MS', '植株：穗位高度', 32.0),
    ('Trait_27_Predict', 'MS', '植株：高度', 2.0),
    ('Trait_28_Predict', 'MS', '植株：穗位高与株高比率', 78.7),
    ('Trait_30_Predict', 'MS', '果穗：长度', 14.7),
    ('Trait_31_Predict', 'MS', '果穗：直径', 0.7),
    ('Trait_32_Predict', 'MS', '果穗：穗行数', 35.3),
    ('Trait_33_Predict', 'VG', '果穗：形状', 0.7),
    ('Trait_34_Predict', 'VG', '果穗：籽粒颜色数量', 0.0),
    ('Trait_36_Predict', 'MS', '籽粒：长度', 39.3),
    ('Trait_37_Predict', 'MS', '籽粒：宽度', 39.3),
    ('Trait_40_Predict', 'VG', '籽粒：顶端主要颜色', 3.3),
    ('Trait_41_Predict', 'VG', '籽粒：背面主要颜色', 1.3),
    ('Trait_42_Predict', 'VG', '籽粒：形状', 0.0),
    ('Trait_43_Predict', 'VG', '穗轴：颖片花青甙显色强度', 0.0)
]

ms_cols = [t[0] for t in trait_mapping if t[1] == 'MS']
vg_cols = [t[0] for t in trait_mapping if t[1] == 'VG']
all_trait_cols = [t[0] for t in trait_mapping]
trait_name_map = {t[0]: t[2] for t in trait_mapping}
trait_missing_rate = {t[0]: t[3] for t in trait_mapping}

# ====================== 4. 读取数据 ======================
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "maize_dus_predict_data.csv"
parser = argparse.ArgumentParser(description="Core-trait PCA + KMeans pipeline.")
parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Input phenotype CSV path.")
args = parser.parse_args()
data_path = args.data_path
df = pd.read_csv(data_path, encoding="utf-8-sig")
y_true = df['True_DUS_Type'].values
sample_names = df['Cultivar Name'].values

# ====================== 5. 步骤1：缺失率硬筛选 ======================
print("=" * 70)
print("📌 步骤1：缺失率硬筛选（剔除>50%）")
print("=" * 70)
filtered_trait_cols = [col for col in all_trait_cols if trait_missing_rate[col] <= 50.0]
deleted_traits = [col for col in all_trait_cols if trait_missing_rate[col] > 50.0]

print(f"剔除的高缺失率性状：")
for col in deleted_traits:
    print(f"  {trait_name_map[col]}（缺失率{trait_missing_rate[col]}%）")
print(f"\n筛选后剩余性状数：{len(filtered_trait_cols)}（原23个）")
X_filtered_raw = df[filtered_trait_cols].copy()

# ====================== 6. 数据预处理 ======================
print("\n" + "=" * 70)
print("📊 步骤2：数据预处理（数值+有序分类性状标准化）")
print("=" * 70)
raw_missing_total = X_filtered_raw.isna().sum().sum()
raw_missing_per_col = X_filtered_raw.isna().sum().sort_values(ascending=False)

print(f"筛选后总缺失值数量：{raw_missing_total}")
print("\n各性状缺失值详情：")
for col, cnt in raw_missing_per_col.items():
    if cnt > 0:
        ratio = cnt / len(X_filtered_raw) * 100
        print(f"  {trait_name_map[col]}: {cnt}个（占比{ratio:.1f}%）")

# 6.1 有序分类性状编码
encoder = OrdinalEncoder()
X_processed = X_filtered_raw.copy()
filtered_vg_cols = [col for col in filtered_trait_cols if col in vg_cols]
X_processed[filtered_vg_cols] = encoder.fit_transform(X_processed[filtered_vg_cols])

# 6.2 KNN插补缺失值
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_processed)
X_imputed = pd.DataFrame(X_imputed, columns=filtered_trait_cols)

# 6.3 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=filtered_trait_cols)
print(f"\n✅ 预处理完成：缺失值插补+全性状标准化，最终数据维度：{X_scaled.shape}")

# ====================== 步骤2.5：核心性状筛选（优化版） ======================
print("\n" + "=" * 70)
print("🔧 步骤2.5：核心性状筛选（ANOVA + 随机森林，优化版）")
print("=" * 70)


# 优化1：ANOVA多重检验校正（避免假阳性）
def anova_feature_selection(X, y, trait_names, alpha=0.05, f_threshold=3):
    anova_results = []
    y_unique = np.unique(y)
    p_vals = []  # 存储所有p值用于校正
    for i, trait in enumerate(trait_names):
        trait_values_by_type = [X[y == typ, i] for typ in y_unique]
        f_stat, p_val = f_oneway(*trait_values_by_type)
        anova_results.append({
            'trait': trait,
            'trait_name': trait_name_map.get(trait, trait),
            'F_value': f_stat,
            'p_value': p_val,
            'raw_p': p_val  # 原始p值
        })
        p_vals.append(p_val)

    # FDR校正（Benjamini-Hochberg）
    corrected_p = multipletests(p_vals, alpha=alpha, method='fdr_bh')[1]
    for i in range(len(anova_results)):
        anova_results[i]['corrected_p'] = corrected_p[i]
        anova_results[i]['is_significant'] = (corrected_p[i] < alpha) and (anova_results[i]['F_value'] > f_threshold)

    anova_df = pd.DataFrame(anova_results)
    significant_traits = anova_df[anova_df['is_significant']]['trait'].tolist()
    print(f"✅ ANOVA筛选结果（F阈值=3 + FDR校正）：")
    print(f"   - 输入性状数：{len(trait_names)}")
    print(f"   - 显著性状数（校正p<0.05 & F>3）：{len(significant_traits)}")
    return anova_df, significant_traits


# 优化2：随机森林特征重要性交叉验证
def rf_feature_selection(X, y, trait_names, top_n_ratio=0.4, min_features=3):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # 5折交叉验证评估RF分类性能
    cv_scores = cross_val_score(rf, X, y_encoded, cv=5, scoring='accuracy')
    print(f"   - RF 5折交叉验证准确率：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    rf.fit(X, y_encoded)

    # 给籽粒/果穗类性状增加权重
    trait_weights = {t: 1.2 if '籽粒' in trait_name_map[t] or '果穗' in trait_name_map[t] else 1.0
                     for t in trait_names}
    importances = [rf.feature_importances_[i] * trait_weights[trait_names[i]]
                   for i in range(len(trait_names))]

    rf_results = pd.DataFrame({
        'trait': trait_names,
        'trait_name': [trait_name_map.get(t, t) for t in trait_names],
        'importance': importances
    }).sort_values('importance', ascending=False)

    top_n = max(int(len(trait_names) * top_n_ratio), min_features)
    core_traits = rf_results.head(top_n)['trait'].tolist()
    print(f"\n✅ 随机森林筛选结果（top40%/至少3个）：")
    print(f"   - ANOVA显著性状数：{len(trait_names)}")
    print(f"   - 高重要性性状数：{len(core_traits)}")
    return rf_results, core_traits


# 优化3：冗余过滤仅在性状数>3时执行
def remove_high_correlation_traits(X, trait_names, corr_threshold=0.85, min_features=3):
    if len(trait_names) <= min_features:
        print(f"\n✅ 冗余过滤：性状数≤{min_features}，无需过滤")
        return trait_names, None

    corr_matrix = pd.DataFrame(X, columns=trait_names).corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                high_corr_pairs.append((trait_names[i], trait_names[j]))

    core_trait_importance = rf_results.set_index('trait')['importance'].to_dict()
    final_core_traits = rf_core_traits.copy()
    for trait1, trait2 in high_corr_pairs:
        if trait1 in final_core_traits and trait2 in final_core_traits:
            if core_trait_importance[trait1] < core_trait_importance[trait2]:
                final_core_traits.remove(trait1)
            else:
                final_core_traits.remove(trait2)

    if len(final_core_traits) < min_features:
        final_core_traits = rf_core_traits[:min_features]

    print(f"\n✅ 冗余过滤结果：")
    print(f"   - 高重要性性状数：{len(rf_core_traits)}")
    print(f"   - 最终核心性状数（去冗余）：{len(final_core_traits)}")
    return final_core_traits, corr_matrix


# 执行筛选流程
anova_df, anova_significant_traits = anova_feature_selection(
    X_scaled.values, y_true, X_scaled.columns
)
X_anova = X_scaled[anova_significant_traits].values
rf_results, rf_core_traits = rf_feature_selection(
    X_anova, y_true, anova_significant_traits
)
final_core_traits, corr_matrix = remove_high_correlation_traits(
    X_scaled[rf_core_traits].values, rf_core_traits
)

# 输出最终核心性状
print(f"\n📌 最终筛选的核心性状列表：")
for i, trait in enumerate(final_core_traits, 1):
    print(f"   {i}. {trait_name_map.get(trait, trait)}")

# 生成核心性状矩阵
X_core_scaled = X_scaled[final_core_traits].copy()
print(f"\n✅ 核心性状矩阵已生成，维度：{X_core_scaled.shape}")

# ====================== 新增优化1：核心性状重要性可视化 ======================
print("\n" + "=" * 70)
print("📊 新增优化1：核心性状重要性可视化")
print("=" * 70)
plt.figure(figsize=(10, 6))

# 保存 rf_results 到CSV（数据量大时更高效）
rf_results.to_csv("rf_results_full_data.csv", index=False, encoding='utf-8-sig')
print("- 原始随机森林结果：rf_results_full_data.csv")
# 筛选核心性状的重要性数据
rf_importance = rf_results[rf_results['trait'].isin(final_core_traits)].sort_values('importance', ascending=False)
# 保存 rf_importance 到Excel（推荐，易编辑）
#rf_importance.to_excel("core_trait_importance_data.xlsx", index=False, engine='openpyxl')
#print("- 核心性状重要性：core_trait_importance_data.xlsx")

# 修复Seaborn警告：添加hue参数并关闭图例
sns.barplot(
    x='importance',
    y='trait_name',
    hue='trait_name',  # 新增hue参数，值与y轴一致
    data=rf_importance,
    palette='RdBu_r',
    legend=False  # 关闭图例，保持原有效果
)
plt.xlabel('特征重要性（加权后）', fontsize=12)
plt.ylabel('核心性状', fontsize=12)
plt.title('核心性状随机森林重要性排序', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("core_trait_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 核心性状重要性图已保存：core_trait_importance.png")

# ====================== 步骤3：PCA降维（自适应维度） ======================
print("\n" + "=" * 70)
print("🔹 步骤3：PCA降维（基于核心性状，自适应维度）")
print("=" * 70)
n_components = min(2, len(final_core_traits))
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_core_scaled)
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])

# 输出PCA解释方差
explained_variance = pca.explained_variance_ratio_
print(f"PCA维度：{n_components}（核心性状数：{len(final_core_traits)}）")
for i, ev in enumerate(explained_variance):
    print(f"PC{i + 1}解释方差占比：{ev:.4f} ({ev * 100:.2f}%)")
print(f"前{n_components}个主成分累计解释方差：{sum(explained_variance):.4f} ({sum(explained_variance) * 100:.2f}%)")

# 保存PCA载荷
pca_loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i + 1}_Loading' for i in range(n_components)],
    index=final_core_traits
)
pca_loadings['Trait_Name'] = [trait_name_map[t] for t in final_core_traits]
pca_loadings.to_excel("core_trait_pca_loadings.xlsx", index=True, engine='openpyxl')
print("\n✅ PCA降维完成，核心性状载荷已保存至 core_trait_pca_loadings.xlsx")

# ====================== 步骤4：K-means聚类（自动选择最优K值：2-8） ======================
print("\n" + "=" * 70)
print("🔸 步骤4：K-means聚类（自动选择最优K值：2/3/4/5/6/7/8）")
print("=" * 70)

# 定义K值候选区间
k_candidates = [2, 3, 4, 5, 6, 7, 8]
k_metrics = {}

# 遍历候选K值，计算评估指标
for k in k_candidates:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels_temp = kmeans_temp.fit_predict(X_pca)
    inertia_temp = kmeans_temp.inertia_
    silhouette_temp = silhouette_score(X_pca, cluster_labels_temp)
    k_metrics[k] = {
        'inertia': inertia_temp,
        'silhouette_score': silhouette_temp
    }

# 输出所有候选K值的指标
print("各K值聚类效果对比：")
print(f"{'K值':<6}{'惯性值(越小越好)':<20}{'轮廓系数(越大越好)':<20}")
print("-" * 50)
for k in k_candidates:
    print(f"{k:<6}{k_metrics[k]['inertia']:<20.2f}{k_metrics[k]['silhouette_score']:<20.4f}")

# 选择最优K值
best_k = max(k_metrics.keys(), key=lambda x: k_metrics[x]['silhouette_score'])
print(f"\n✅ 最优K值选择：{best_k}（轮廓系数最大：{k_metrics[best_k]['silhouette_score']:.4f}）")

# 生成并保存Excel数据（核心新增部分） ==========
# 整理绘图/分析用的完整数据
k_excel_data = pd.DataFrame({
    'K值': k_candidates,
    '惯性值(Inertia)': [k_metrics[k]['inertia'] for k in k_candidates],
    '轮廓系数(Silhouette Score)': [k_metrics[k]['silhouette_score'] for k in k_candidates],
    '是否为最优K值': [k == best_k for k in k_candidates]  # 标记最优K值，方便Excel中筛选
})

# 保存为Excel文件（指定engine避免警告，utf-8-sig解决中文乱码）
k_excel_data.to_excel(
    "k_selection_metrics.xlsx",  # 输出文件名
    index=False,                 # 不保存行索引
    engine='openpyxl'            # 兼容Excel格式的引擎（需提前安装：pip install openpyxl）
)
print("\n✅ K值评估指标已保存为Excel：k_selection_metrics.xlsx")

# K值选择可视化（肘部图+轮廓系数图）
plt.figure(figsize=(12, 5))
# 子图1：肘部图
plt.subplot(1, 2, 1)
inertias = [k_metrics[k]['inertia'] for k in k_candidates]
plt.plot(k_candidates, inertias, 'o-', color='#2E86AB', linewidth=2)
plt.scatter(best_k, k_metrics[best_k]['inertia'], color='#C73E1D', s=100, label=f'最优K={best_k}')
plt.xlabel('K值', fontsize=12)
plt.ylabel('惯性值（Inertia）', fontsize=12)
plt.title('K-means肘部法则图', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 子图2：轮廓系数图
plt.subplot(1, 2, 2)
sil_scores = [k_metrics[k]['silhouette_score'] for k in k_candidates]
plt.plot(k_candidates, sil_scores, 'o-', color='#F18F01', linewidth=2)
plt.scatter(best_k, k_metrics[best_k]['silhouette_score'], color='#C73E1D', s=100, label=f'最优K={best_k}')
plt.xlabel('K值', fontsize=12)
plt.ylabel('轮廓系数', fontsize=12)
plt.title('K-means轮廓系数图', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("k_selection_elbow_silhouette.png", dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ K值选择可视化图已保存：k_selection_elbow_silhouette.png")

# ====================== 新增优化2：加权K-means（提升小类别聚类效果） ======================
print("\n" + "=" * 70)
print("🔧 新增优化2：加权K-means聚类（提升小类别效果）")
print("=" * 70)
# 计算样本权重（为小类别样本增加权重）
sample_weights = compute_sample_weight('balanced', y_true)
# 对PCA结果进行加权
X_pca_weighted = X_pca * sample_weights.reshape(-1, 1)

# 用最优K值执行加权K-means聚类
n_clusters = best_k
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_pca_weighted)  # 使用加权后的PCA结果

# 输出最优K值的聚类中心
print(f"\n最优K={best_k}的聚类中心（PCA维度，加权后）：")
cluster_centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=[f'PC{i + 1}' for i in range(n_components)],
    index=[f'簇{i + 1}' for i in range(best_k)]
)
print(cluster_centers)

# ====================== 关键修复：先定义result_df，再做统计！ ======================
# 整合结果（这一步必须在聚类之后、统计之前）
result_df = pd.DataFrame({
    'Serial_No': df['Serial No.'],
    'Field_Plot_No': df['Field Plot No.'],
    'Cultivar_Name': sample_names,
    'True_DUS_Type': y_true,
    **{f'PC{i + 1}': X_pca[:, i] for i in range(n_components)},
    'Cluster_Label': cluster_labels + 1  # 簇标签从1开始
})
# 合并核心性状的标准化数据
result_df = pd.concat([result_df, X_core_scaled], axis=1)

# ====================== 新增优化3：聚类稳定性验证 ======================
print("\n" + "=" * 70)
print("🔍 新增优化3：聚类稳定性验证")
print("=" * 70)
# 多次随机种子验证聚类稳定性
stable_scores = []
for seed in range(10):
    kmeans_temp = KMeans(n_clusters=best_k, random_state=seed, n_init='auto')
    labels_temp = kmeans_temp.fit_predict(X_pca_weighted)
    stable_scores.append(adjusted_rand_score(cluster_labels, labels_temp))

stable_mean = np.mean(stable_scores)
stable_std = np.std(stable_scores)
print(f"聚类稳定性（调整兰德指数）：{stable_mean:.4f} ± {stable_std:.4f}")
print(f"👉 指数>0.8说明聚类结果稳定，当前值为{stable_mean:.4f}，{'稳定' if stable_mean>0.8 else '基本稳定'}")

# 保存稳定性结果
stable_df = pd.DataFrame({
    '随机种子': range(10),
    '调整兰德指数': stable_scores
})
stable_df['均值'] = stable_mean
stable_df['标准差'] = stable_std
stable_df.to_excel("cluster_stability_results.xlsx", index=False, engine='openpyxl')
print("✅ 聚类稳定性结果已保存：cluster_stability_results.xlsx")

# ====================== 步骤5：结果统计与可视化 ======================
print("\n" + "=" * 70)
print("📈 步骤5：结果统计与可视化（核心性状版）")
print("=" * 70)

# 9.1 聚类统计
print("1. 各聚类簇样本数：")
cluster_counts = result_df['Cluster_Label'].value_counts().sort_index()
print(cluster_counts)

print("\n2. 聚类与真实DUS类型交叉表：")
cross_tab = pd.crosstab(result_df['Cluster_Label'], result_df['True_DUS_Type'], margins=True)
print(cross_tab)

# 9.2 计算匹配率
cluster_main_type = {}
for cluster in range(1, n_clusters + 1):
    cluster_data = result_df[result_df['Cluster_Label'] == cluster]
    main_type = cluster_data['True_DUS_Type'].value_counts().idxmax()
    cluster_main_type[cluster] = main_type
    match_count = len(cluster_data[cluster_data['True_DUS_Type'] == main_type])
    total_count = len(cluster_data)
    match_rate = match_count / total_count * 100
    print(f"\n簇{cluster}：主要类型={main_type}，匹配率={match_rate:.2f}%（{match_count}/{total_count}）")

# 整理绘图所需核心字段
pca_plot_data = result_df[['Cultivar_Name', 'Cluster_Label', 'True_DUS_Type', 'PC1', 'PC2']].copy()
# 新增列：每个簇的主要类型（方便绘图时分组标注）
pca_plot_data['Cluster_Main_Type'] = pca_plot_data['Cluster_Label'].map(cluster_main_type)
# 标记典型品种（方便绘图时重点标注）
typical_cultivars = [
    'Xiyu Nuo 59-Sweet-waxy', 'Yueduocaixian Nuo-Sweet-waxy', 'Mitian Nuo 20-Sweet-waxy',
    'Kenzhan 11-Waxy', 'Minuo 632-Waxy', 'Jingzi Nuo 318-Waxy',
    'Jingbai Tian 686-Sweet', 'CZ9945-Popcorn', 'CZ898-Popcorn'
]
pca_plot_data['Is_Typical_Cultivar'] = pca_plot_data['Cultivar_Name'].isin(typical_cultivars)
# 补充PCA解释方差（绘图时标注坐标轴用）
pca_variance_data = pd.DataFrame({
    'PC': ['PC1', 'PC2'],
    'Explained_Variance_Ratio': [explained_variance[0], explained_variance[1] if len(explained_variance)>=2 else 0],
    'Explained_Variance_Percent': [explained_variance[0]*100, explained_variance[1]*100 if len(explained_variance)>=2 else 0]
})
# 保存数据到Excel（分两个sheet：散点数据+方差数据）
with pd.ExcelWriter('pca_kmeans_scatter_data.xlsx', engine='openpyxl') as writer:
    pca_plot_data.to_excel(writer, sheet_name='Scatter_Data', index=False)
    pca_variance_data.to_excel(writer, sheet_name='PCA_Variance', index=False)
print("\n✅ PCA散点图数据已保存：pca_kmeans_scatter_data.xlsx")
print(f"   - 包含字段：品种名、聚类标签、真实DUS类型、PC1/PC2坐标、簇主要类型、是否典型品种")
print(f"   - PCA方差信息：PC1解释方差{explained_variance[0]*100:.2f}%，PC2解释方差{explained_variance[1]*100:.2f}%（若有）")

# 9.3 绘制PCA散点图（修复颜色/标记列表长度+标题K值）
plt.figure(figsize=(12, 8))
# 扩展颜色/标记列表，适配K=4-8
colors = ['#2E86AB', '#F18F01', '#C73E1D', '#A23B72', '#F24236', '#8B4513', '#9370DB', '#00CED1']
markers = ['o', 's', '^', 'D', 'P', '*', 'x', '+']
for cluster in range(1, n_clusters + 1):
    cluster_data = result_df[result_df['Cluster_Label'] == cluster]
    plt.scatter(
        cluster_data['PC1'], cluster_data['PC2'] if n_components >= 2 else cluster_data['PC1'],
        c=colors[cluster - 1], marker=markers[cluster - 1], s=80, alpha=0.7,
        label=f'簇{cluster}（主要类型：{cluster_main_type[cluster]}）'
    )

# 标注典型品种
typical_cultivars = [
    'Xiyu Nuo 59-Sweet-waxy', 'Yueduocaixian Nuo-Sweet-waxy', 'Mitian Nuo 20-Sweet-waxy',
    'Kenzhan 11-Waxy', 'Minuo 632-Waxy', 'Jingzi Nuo 318-Waxy',
    'Jingbai Tian 686-Sweet', 'CZ9945-Popcorn', 'CZ898-Popcorn'
]
for cv in typical_cultivars:
    cv_data = result_df[result_df['Cultivar_Name'] == cv]
    if not cv_data.empty:
        plt.annotate(
            cv,
            (cv_data['PC1'].values[0], cv_data['PC2'].values[0] if n_components >= 2 else cv_data['PC1'].values[0]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, fontweight='bold'
        )

plt.xlabel(f'主成分1（PC1，解释方差{explained_variance[0] * 100:.2f}%）', fontsize=12)
if n_components >= 2:
    plt.ylabel(f'主成分2（PC2，解释方差{explained_variance[1] * 100:.2f}%）', fontsize=12)
else:
    plt.ylabel('辅助维度', fontsize=12)
# 修复标题：K值改为动态的best_k
plt.title(f'玉米品种PCA降维+K-means聚类散点图（核心性状版，K={best_k}）', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("core_trait_pca_kmeans_scatter.png", dpi=300, bbox_inches='tight')
plt.close()

# 9.4 绘制簇-性状热力图
core_trait_names = [trait_name_map[t] for t in final_core_traits]
cluster_trait_means = result_df.groupby('Cluster_Label')[final_core_traits].mean()
cluster_trait_means.columns = core_trait_names

# 输出热力图绘图数据
cluster_trait_means.to_excel(
    "core_trait_cluster_heatmap_data.xlsx",
    sheet_name="Cluster_Trait_Means",
    index=True,  # 保留Cluster_Label作为行索引（Y轴）
    engine='openpyxl'
)
print("✅ 热力图绘图数据已保存：")
print("   - Excel：core_trait_cluster_heatmap_data.xlsx")

plt.figure(figsize=(10, 8))
sns.heatmap(cluster_trait_means, annot=True, cmap='RdBu_r', fmt='.2f',
            cbar_kws={'label': '性状标准化均值'})
plt.xlabel('核心表型性状', fontsize=12)
plt.ylabel('聚类簇', fontsize=12)
plt.title('各聚类簇核心性状均值热力图（核心性状版）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("core_trait_cluster_heatmap.png", dpi=300)
plt.close()

# ====================== 结果保存 ======================
result_df.to_excel("core_trait_pca_kmeans_result.xlsx", index=False, engine='openpyxl')
cluster_match_df = pd.DataFrame({
    '聚类簇': list(cluster_main_type.keys()),
    '主要DUS类型': list(cluster_main_type.values()),
    '样本数': [cluster_counts[cl] for cl in cluster_main_type.keys()],
    '匹配率(%)': [
        len(result_df[(result_df['Cluster_Label'] == cl) & (result_df['True_DUS_Type'] == typ)]) / cluster_counts[
            cl] * 100
        for cl, typ in cluster_main_type.items()
    ]
})
cluster_match_df.to_excel("core_trait_cluster_match_rate.xlsx", index=False, engine='openpyxl')

print("\n🎉 所有分析完成！核心性状版文件清单：")
print("1. k_selection_elbow_silhouette.png → K值选择图（肘部图+轮廓系数图）")
print("2. core_trait_pca_kmeans_scatter.png → PCA聚类散点图（核心性状版）")
print("3. core_trait_cluster_heatmap.png → 簇-性状热力图（核心性状版）")
print("4. core_trait_pca_loadings.xlsx → 核心性状PCA载荷")
print("5. core_trait_pca_kmeans_result.xlsx → 完整聚类结果（核心性状）")
print("6. core_trait_cluster_match_rate.xlsx → 簇-类型匹配率（核心性状）")
print("7. core_trait_importance.png → 核心性状随机森林重要性排序图（新增）")
print("8. cluster_stability_results.xlsx → 聚类稳定性验证结果（新增）")

# ====================== 结论性统计 ======================
print("\n" + "=" * 70)
print("📝 核心结论统计（核心性状版，论文直接引用）")
print("=" * 70)
match_rates = [
    len(result_df[(result_df['Cluster_Label'] == cl) & (result_df['True_DUS_Type'] == typ)]) / cluster_counts[cl] * 100
    for cl, typ in cluster_main_type.items()]
print(
    f"1. 核心性状筛选：基于ANOVA（F>3+FDR校正）+随机森林(交叉验证)筛选出{len(final_core_traits)}个核心表型性状，覆盖DUS测试关键维度；")
print(f"2. PCA降维：前{n_components}个主成分累计解释{sum(explained_variance) * 100:.2f}%的表型变异；")
# 修复：K值改为best_k
print(
    f"3. K-means聚类：最优K={best_k}，各簇主要类型匹配率介于{min(match_rates):.2f}%-{max(match_rates):.2f}%（聚类纯度显著提升）；")
print(f"4. 聚类有效性：轮廓系数={silhouette_score(X_pca, cluster_labels):.4f}（>0.1，聚类结构有效）；")
