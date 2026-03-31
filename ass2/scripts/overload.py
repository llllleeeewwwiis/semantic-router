import psutil
import time
import os

def monitor_total_cpu():
    print(f"开始监控 CPU 利用率 (按 Ctrl+C 停止)...核心数: {psutil.cpu_count()}")
    print(f"{'时间':<10} {'总利用率 (SUM %)' :<20} {'各核详情'}")
    print("-" * 60)
    
    try:
        while True:
            # percpu=True 返回每个核心的利用率列表
            per_cpu = psutil.cpu_percent(interval=1, percpu=True)
            total_sum = sum(per_cpu)
            
            # 格式化各核数据，便于观察是否只有单核在忙
            cores_str = " | ".join([f"{v:>5}%" for v in per_cpu])
            
            print(f"{time.strftime('%H:%M:%S'):<10} {total_sum:>12.1f}%      [{cores_str}]")
            
    except KeyboardInterrupt:
        print("\n监控结束。")

if __name__ == "__main__":
    monitor_total_cpu()
