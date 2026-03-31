import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 整理新跑的数据
data = {
    'Concurrency': [1, 5, 10, 20, 50],
    'QPS': [2.0, 3.1, 3.2, 3.5, 5.3],
    'p50': [433.0, 1457.0, 3117.5, 5714.2, 7884.2],
    'p99': [1583.4, 4773.9, 7386.2, 10299.2, 10743.6],
    'Errors': [0, 0, 0, 18, 204],
    'CPU': [31.2, 73.3, 71.4, 68.9, 78.3]
}
df = pd.DataFrame(data)

# 2. 风格配置
plt.rcParams['font.sans-serif'] = ['Arial']
color_qps = '#A66133'     # 棕橙色
color_cpu = '#2C5F2D'     # 深绿色 (代表资源占用)
color_p50 = '#A66133'    # 棕橙色
color_p99 = '#5D4A87'    # 深紫色
color_err = '#D62728'    # 红色 (警告)
grid_style = dict(linestyle='--', color='#E0E0E0', alpha=0.7)

def set_square_plot(ax, title):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Concurrency', fontsize=11)
    ax.grid(True, **grid_style)
    ax.spines['top'].set_visible(False)

# --- 图表 1: QPS & CPU 利用率 (双轴正方形) ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
ax1_twin = ax1.twinx()

ln1 = ax1.plot(df['Concurrency'], df['QPS'], color=color_qps, marker='o', linewidth=2.5, label='QPS')
ln2 = ax1_twin.plot(df['Concurrency'], df['CPU'], color=color_cpu, marker='D', linewidth=2.5, label='CPU (%)')

ax1.set_ylabel('QPS', color=color_qps, fontweight='bold')
ax1_twin.set_ylabel('CPU Utilization (%)', color=color_cpu, fontweight='bold')
ax1_twin.set_ylim(0, 100) # CPU 通常看 0-100%

set_square_plot(ax1, 'System Throughput & CPU Load')
# 合并图例
lns = ln1 + ln2
ax1.legend(lns, [l.get_label() for l in lns], loc='lower right')

# --- 图表 2: p50 & p99 Latency (正方形) ---
fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.plot(df['Concurrency'], df['p50'], color=color_p50, marker='o', linewidth=2.5, label='p50 Latency')
ax2.plot(df['Concurrency'], df['p99'], color=color_p99, marker='^', linewidth=2.5, label='p99 Latency')

ax2.set_ylabel('Latency (ms)', fontweight='bold')
set_square_plot(ax2, 'Latency Trends')
ax2.legend(loc='upper left')

# --- 图表 3: Errors (正方形) ---
fig3, ax3 = plt.subplots(figsize=(7, 7))
ax3.plot(df['Concurrency'], df['Errors'], color=color_err, marker='s', linewidth=2.5, label='Errors')
ax3.fill_between(df['Concurrency'], df['Errors'], color=color_err, alpha=0.1) # 增加阴影强调错误区域

ax3.set_ylabel('Total Errors', color=color_err, fontweight='bold')
set_square_plot(ax3, 'Request Errors')
ax3.legend(loc='upper left')

# 3. 终端统计简报
print("="*45)
print(f"{'Concurrency':<12} | {'QPS':<6} | {'CPU%':<6} | {'p99(ms)':<8}")
print("-" * 45)
for _, row in df.iterrows():
    print(f"{int(row['Concurrency']):<12} | {row['QPS']:<6.1f} | {row['CPU']:<6.1f} | {row['p99']:<8.1f}")
print("="*45)

plt.tight_layout()
plt.show()