import json

def process_json_data(input_file, output_file):
    try:
        # 1. 读取原始 JSON 数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 遍历列表处理每一个字典对象
        for item in data:
            # 去掉 'label' 字段
            item.pop('label', None)
            
            # 将 'source_id' 重命名为 'question_id'
            # 如果 source_id 存在，pop 会返回它的值并删除该键，然后赋给新键
            if 'source_id' in item:
                item['question_id'] = item.pop('source_id')

        # 3. 将修改后的数据保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成！'label' 已移除，'source_id' 已更名为 'question_id'。")
        print(f"结果已保存至: {output_file}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 执行脚本
process_json_data('routing_accuracy_holdout_results.json', 'data_cleaned.json')