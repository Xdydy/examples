# 打开文件并读取内容
file_path = "input_device.txt"  # 替换为你的文件路径
table_info = {}
current_table_name = None

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) > 0:
            if not parts[0].isnumeric():
                continue
            if current_table_name == None:
                table_number = int(parts[0])
                if len(parts) >= 2:
                    current_table_name = " ".join(parts[1:])
                    table_info[current_table_name] = []
            elif current_table_name and len(parts) >= 2:
                table_info[current_table_name].append(" ".join(parts[1:]))
        else:
            current_table_name = None

# 打印表格名和属性名
output_path="output_device.txt"
with open(output_path, 'w', encoding='utf-8') as file:
    for table_name, attributes in table_info.items():
        file.write("表格名: "+ table_name + "\n")
        # file.write("属性名: "+ str(attributes) + "\n")
        file.write("\n")
