import pickle
import os

file_path = os.path.join('processed_data/test.pt')  # 假设读取 train.pt 文件

with open(file_path, 'rb') as f:
    questions = pickle.load(f)
    sparqls = pickle.load(f)
    choices = pickle.load(f)
    answers = pickle.load(f)

# 检查数据形状
print(questions.shape)  # 输出 (128, 50)
print(sparqls.shape)    # 输出 (128, 70)
print(choices.shape)    # 输出 (128, 4)
print(answers.shape)    # 输出 (128,)
