import json

# 打开 JSON 文件并读取
with open("data.json", "r") as file:
    data = json.load(file)

# 输出读取到的数据
print(data)
