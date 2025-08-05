import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

# 读取CSV文件，不包含表头
file_path = r"D:\pythonproject\STCA-LLM\STCA-LLM_stage3\dataset\wind speed of 20 turbines.csv"
df = pd.read_csv(file_path, header=None)

# 将所有列的数据类型转换为浮点数，并处理异常数据
df = df.apply(pd.to_numeric, errors='coerce')

# 初始化一个20x20的矩阵
dtw_matrix = [[0 for _ in range(20)] for _ in range(20)]

# 使用tqdm添加进度条
for i in tqdm(range(20), desc="Calculating DTW"):
    for j in range(i, 20):
        distance, _ = fastdtw(df.iloc[:, i], df.iloc[:, j], dist=2)
        dtw_matrix[i][j] = dtw_matrix[j][i] = distance
# 显示DTW矩阵
dtw_matrix_df = pd.DataFrame(dtw_matrix)
print(dtw_matrix_df)

# 保存DTW矩阵为CSV文件
save_path = r"D:\pythonproject\STCA-LLM\DTW.csv"  # 修正文件保存路径
dtw_matrix_df.to_csv(save_path, index=False, header=False)

print("DTW matrix saved to:", save_path)
