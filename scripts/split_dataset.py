import json
import random

# 读取数据
with open('dataset/non-vulnerables.json', 'r', encoding='utf-8') as f:
    non_vul = json.load(f)
with open('dataset/vulnerables.json', 'r', encoding='utf-8') as f:
    vul = json.load(f)

# 给每个样本加标签
for item in non_vul:
    item['label'] = 0
for item in vul:
    item['label'] = 1

# 合并并打乱
all_data = non_vul + vul
random.shuffle(all_data)

# 拆分
n = len(all_data)
train = all_data[:int(0.8*n)]
valid = all_data[int(0.8*n):int(0.9*n)]
test  = all_data[int(0.9*n):]

# 保存为 jsonl
def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

save_jsonl(train, 'dataset/train.jsonl')
save_jsonl(valid, 'dataset/valid.jsonl')
save_jsonl(test,  'dataset/test.jsonl')