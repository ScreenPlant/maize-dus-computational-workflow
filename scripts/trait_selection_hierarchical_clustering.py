import argparse
from pathlib import Path
# ====================== 1. 导入所有依赖库 ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier

# ====================== 2. 基础全局设置 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 3. 关键变量定义 ======================
# 【核心】23个性状映射（含缺失率标注）
trait_mapping = [
    ('Trait_1_Predict', 'VG', '幼苗：第一叶顶端形状', 38.7),
    ('Trait_2_Predict', 'VG', '幼苗：第一叶鞘花青甙显色强度', 0.0),
    ('Trait_3_Predict', 'VG', '叶片：绿色程度', 0.0),
    ('Trait_6_Predict', 'VG', '植株：上部叶片与茎秆夹角', 0.0),
    ('Trait_7_Predict', 'VG', '叶片：弯曲程度', 0.0),
    ('Trait_9_Predict', 'VG', '雌穗：花丝花青甙显色强度', 61.3),  # 高缺失
    ('Trait_10_Predict', 'VG', '雄穗：花药花青甙显色强度', 39.3),
    ('Trait_23_Predict', 'MS', '叶片：长度', 0.0),
    ('Trait_24_Predict', 'MS', '叶片：宽度', 34.7),
    ('Trait_26_Predict', 'MS', '植株：穗位高度', 32.0),
    ('Trait_27_Predict', 'MS', '植株：高度', 2.0),
    ('Trait_28_Predict', 'MS', '植株：穗位高与株高比率', 78.7),  # 高缺失
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

# 拆分变量
ms_cols = [t[0] for t in trait_mapping if t[1] == 'MS']
vg_cols = [t[0] for t in trait_mapping if t[1] == 'VG']
all_trait_cols = [t[0] for t in trait_mapping]
trait_name_map = {t[0]: t[2] for t in trait_mapping}
trait_missing_rate = {t[0]: t[3] for t in trait_mapping}  # 缺失率映射

# ====================== 4. 读取数据 ======================
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "maize_dus_predict_data.csv"
parser = argparse.ArgumentParser(description="Trait selection and weighted hierarchical clustering pipeline.")
parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Input phenotype CSV path.")
args = parser.parse_args()
data_path = args.data_path
df = pd.read_csv(data_path, encoding="utf-8-sig")

# 提取关键列
y_true = df['True_DUS_Type'].values
sample_names = df['Cultivar Name'].values

# ====================== 5. 步骤1：缺失率硬筛选（剔除>50%） ======================
print("="*70)
print("📌 步骤1：缺失率硬筛选（剔除>50%）")
print("="*70)
# 筛选缺失率≤50%的性状
filtered_trait_cols = [col for col in all_trait_cols if trait_missing_rate[col] <= 50.0]
deleted_traits = [col for col in all_trait_cols if trait_missing_rate[col] > 50.0]

print(f"剔除的高缺失率性状：")
for col in deleted_traits:
    print(f"  {trait_name_map[col]}（缺失率{trait_missing_rate[col]}%）")
print(f"\n筛选后剩余性状数：{len(filtered_trait_cols)}（原23个）")

# 提取筛选后的原始性状数据
X_filtered_raw = df[filtered_trait_cols].copy()

# ====================== 6. 缺失值处理（仅针对筛选后性状） ======================
print("\n" + "="*70)
print("📊 筛选后性状缺失值统计（插补前）")
print("="*70)
raw_missing_total = X_filtered_raw.isna().sum().sum()
raw_missing_per_col = X_filtered_raw.isna().sum().sort_values(ascending=False)

print(f"筛选后总缺失值数量：{raw_missing_total}")
print("\n各性状缺失值详情：")
for col, cnt in raw_missing_per_col.items():
    if cnt > 0:
        ratio = cnt / len(X_filtered_raw) * 100
        print(f"  {trait_name_map[col]}: {cnt}个（占比{ratio:.1f}%）")

# 有序分类性状编码
encoder = OrdinalEncoder()
X_processed = X_filtered_raw.copy()
filtered_vg_cols = [col for col in filtered_trait_cols if col in vg_cols]
X_processed[filtered_vg_cols] = encoder.fit_transform(X_processed[filtered_vg_cols])

# KNN插补
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_processed)
X_imputed = pd.DataFrame(X_imputed, columns=filtered_trait_cols)

# 数值性状标准化
filtered_ms_cols = [col for col in filtered_trait_cols if col in ms_cols]
scaler = StandardScaler()
X_imputed[filtered_ms_cols] = scaler.fit_transform(X_imputed[filtered_ms_cols])

print(f"\n✅ 缺失值插补完成！插补后缺失值数：{X_imputed.isna().sum().sum()}")

# ====================== 7. 步骤2+3：量化验证（ANOVA+随机森林+去冗余） ======================
print("\n" + "="*70)
print("🔍 步骤2+3：量化验证（ANOVA+随机森林+去冗余）")
print("="*70)

# 7.1 ANOVA：筛选组间差异显著的性状（F>5，P<0.05）
anova_results = {}
for trait in filtered_trait_cols:
    groups = [X_imputed.loc[y_true == typ, trait].values for typ in np.unique(y_true)]
    f_val, p_val = f_oneway(*groups)
    anova_results[trait] = {'F值': f_val, 'P值': p_val, '显著': p_val < 0.05 and f_val > 5}

# 筛选ANOVA显著的性状
anova_significant_cols = [col for col in filtered_trait_cols if anova_results[col]['显著']]
print(f"\nANOVA组间差异显著的性状（F>5，P<0.05）：{len(anova_significant_cols)}个")
for col in anova_significant_cols:
    print(f"  {trait_name_map[col]}: F={anova_results[col]['F值']:.2f}, P={anova_results[col]['P值']:.4f}")

# 7.2 随机森林特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_imputed, y_true)

# 提取重要性并排序
feature_importance = pd.DataFrame({
    'Trait': filtered_trait_cols,
    'Importance': rf.feature_importances_,
    'Trait_Name': [trait_name_map[t] for t in filtered_trait_cols],
    'Missing_Rate': [trait_missing_rate[t] for t in filtered_trait_cols]
}).sort_values('Importance', ascending=False)

# 可视化重要性（保存图片）
plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Trait_Name'][:15], feature_importance['Importance'][:15])
plt.xlabel('特征重要性得分')
plt.ylabel('性状名称')
plt.title('玉米核心性状重要性排序（随机森林）')
plt.tight_layout()
plt.savefig("trait_importance.png", dpi=300)
plt.close()

# 7.3 去冗余（剔除相关系数>0.8的性状）
corr_matrix = X_imputed.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

print(f"\n高相关性状对（|r|>0.8，需剔除冗余）：")
for t1, t2, corr in high_corr_pairs:
    print(f"  {trait_name_map[t1]} ↔ {trait_name_map[t2]}: r={corr:.2f}")

# 7.4 最终核心性状筛选（ANOVA显著 + 重要性前12 + 去冗余）
# 取重要性前12个 + ANOVA显著的交集
top_importance_cols = feature_importance['Trait'].head(12).tolist()
final_core_traits = list(set(top_importance_cols) & set(anova_significant_cols))

# 手动保留DUS核心性状（兜底）
dus_core_traits = [
    # 'Trait_27_Predict',  # 株高
    # 'Trait_30_Predict',  # 果穗长
    # 'Trait_31_Predict',  # 果穗直径
    # 'Trait_36_Predict',  # 籽粒长
    # 'Trait_37_Predict',  # 籽粒宽
    # 'Trait_40_Predict'   # 籽粒顶端颜色
]
for core_trait in dus_core_traits:
    if core_trait not in final_core_traits:
        final_core_traits.append(core_trait)

# 去重并排序
final_core_traits = list(set(final_core_traits))
final_core_traits.sort(key=lambda x: feature_importance.loc[feature_importance['Trait']==x, 'Importance'].values[0], reverse=True)

print(f"\n🎯 最终筛选的核心性状（{len(final_core_traits)}个）：")
for i, col in enumerate(final_core_traits, 1):
    imp_score = feature_importance.loc[feature_importance['Trait']==col, 'Importance'].values[0]
    print(f"  {i}. {trait_name_map[col]}（重要性得分：{imp_score:.3f}，缺失率：{trait_missing_rate[col]}%）")

# 提取核心性状的插补后数据
X_core = X_imputed[final_core_traits].copy()

# ====================== 8. 样本不平衡处理 + 加权聚类 ======================
print("\n" + "="*70)
print("⚖️  基于核心性状的加权层次聚类")
print("="*70)

# 计算权重
class_counts = pd.Series(y_true).value_counts()
max_count = class_counts.max()
class_weights_dict = (max_count / class_counts).to_dict()
weights_raw = np.array([class_weights_dict[typ] for typ in y_true])
sample_weights = (weights_raw - weights_raw.min()) / (weights_raw.max() - weights_raw.min()) * 9 + 1

# 加权特征矩阵 + 压缩距离矩阵
X_weighted = X_core.values * sample_weights[:, np.newaxis]
compressed_dist = pdist(X_weighted, metric='euclidean')

# Ward法聚类
linkage_matrix = linkage(compressed_dist, method='ward')
n_clusters = 5
cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

# ====================== 9. 结果可视化与保存 ======================
# 9.1 聚类树状图
plt.figure(figsize=(20, 10))
dendrogram(
    linkage_matrix,
    labels=sample_names,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=linkage_matrix[:, 2].max() * 0.7
)
plt.title('玉米品种核心性状加权层次聚类树状图（筛选后{}个性状）'.format(len(final_core_traits)), fontsize=16)
plt.xlabel('品种名称', fontsize=12)
plt.ylabel('加权欧式距离', fontsize=12)
plt.tight_layout()
plt.savefig("maize_core_trait_clustering.png", dpi=300, bbox_inches='tight')
plt.close()

# 9.2 整合结果
result_df = pd.DataFrame({
    'Serial_No': df['Serial No.'],
    'Field_Plot_No': df['Field Plot No.'],
    'Cultivar_Name': sample_names,
    'True_DUS_Type': y_true,
    'Sample_Weight': sample_weights,
    'Cluster_Label': cluster_labels
})
# 合并核心性状数据
result_df = pd.concat([result_df, X_core], axis=1)

# 9.3 统计结果
print("\n📊 核心性状聚类结果统计：")
print("1. 各聚类簇样本数：")
print(result_df['Cluster_Label'].value_counts().sort_index())
print("\n2. 聚类与真实品种类型交叉表：")
print(pd.crosstab(result_df['Cluster_Label'], result_df['True_DUS_Type'], margins=True))

# 9.4 保存结果
result_df.to_excel("maize_core_trait_clustering_result.xlsx", index=False, engine='openpyxl')
# 保存核心性状列表
core_trait_list = pd.DataFrame({
    '性状列名': final_core_traits,
    '性状名称': [trait_name_map[t] for t in final_core_traits],
    '缺失率(%)': [trait_missing_rate[t] for t in final_core_traits],
    '重要性得分': [feature_importance.loc[feature_importance['Trait']==t, 'Importance'].values[0] for t in final_core_traits]
})
core_trait_list.to_excel("maize_core_trait_list.xlsx", index=False, engine='openpyxl')

# ====================== 10. 运行完成 ======================
print("\n🎉 所有步骤执行完成！生成文件：")
print("1. trait_importance.png → 性状重要性排序图（论文可用）")
print("2. maize_core_trait_clustering.png → 核心性状聚类树状图")
print("3. maize_core_trait_clustering_result.xlsx → 核心性状聚类结果表")
print("4. maize_core_trait_list.xlsx → 核心性状清单（含缺失率/重要性）")
