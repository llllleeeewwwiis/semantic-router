import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
data = [
    ["science", "biology", 52.4, 26.7, 78],
    ["science", "chemistry", 33.9, 14.3, 76],
    ["science", "health", 64.5, 46.2, 75],
    ["stem", "engineering", 30.6, 28.6, 76],
    ["stem", "math", 40.9, 30.0, 76],
    ["stem", "physics", 42.4, 12.5, 75],
    ["cs", "computer science", 54.4, 11.1, 77],
    ["humanities", "history", 61.3, 0.0, 78],
    ["humanities", "law", 41.3, 33.3, 78],
    ["humanities", "philosophy", 55.4, 50.0, 78],
    ["humanities", "psychology", 73.3, 44.4, 78],
    ["business", "business", 37.3, 10.0, 77],
    ["business", "economics", 60.6, 16.7, 77],
    ["default", "other", 12.3, 30.0, 75],
]

df = pd.DataFrame(data, columns=['Group', 'Subject', 'Correct_Rate', 'Incorrect_Rate', 'Samples'])

# 2. 计算准确率差值 (Gap)
df['Gap'] = df['Correct_Rate'] - df['Incorrect_Rate']

# 按学科群聚合
group_df = df.groupby('Group').agg({
    'Correct_Rate': 'mean',
    'Incorrect_Rate': 'mean',
    'Gap': 'mean'
}).reset_index()

# 计算总体
total_correct = df['Correct_Rate'].mean()
total_incorrect = df['Incorrect_Rate'].mean()
total_gap = total_correct - total_incorrect

# 3. 终端输出分析结果
print("="*50)
print(f"{'学科/学科群':<20} | {'准确率差值 (Correct - Incorrect)':<30}")
print("-"*50)
for _, row in df.iterrows():
    print(f"{row['Subject']:<20} | {row['Gap']:>8.1f}%")
print("-"*50)
for _, row in group_df.iterrows():
    print(f"Group: {row['Group']:<13} | {row['Gap']:>8.1f}%")
print("-"*50)
print(f"{'总体平均':<20} | {total_gap:>8.1f}%")
print("="*50)

# 4. 绘图设置
plt.rcParams['font.sans-serif'] = ['Arial'] 
color_correct = '#A66133' # 参考图中的棕橙色
color_incorrect = '#5D4A87' # 参考图中的深紫色

def plot_bars(ax, labels, val1, val2, title, is_square=False):
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, val1, width, label='Routing Correct', color=color_correct)
    rects2 = ax.bar(x + width/2, val2, width, label='Routing Incorrect', color=color_incorrect)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if not is_square else 0)
    ax.legend()
    ax.grid(axis='y', linestyle='-', alpha=0.7)
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)

# 图表 1：学科群（正方形）
fig1, ax1 = plt.subplots(figsize=(7, 7))
plot_bars(ax1, group_df['Group'], group_df['Correct_Rate'], group_df['Incorrect_Rate'], 'Accuracy by Subject Group', is_square=True)
plt.tight_layout()

# 图表 2：每个学科（长方形）
fig2, ax2 = plt.subplots(figsize=(12, 6))
plot_bars(ax2, df['Subject'], df['Correct_Rate'], df['Incorrect_Rate'], 'Accuracy by Specific Subject')
plt.tight_layout()

plt.show()
