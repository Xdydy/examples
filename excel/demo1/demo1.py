import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('example.xlsx')

# 显示前几行数据
print(df.head())

# 访问特定列的数据
column_data = df['Column_Name']
print(column_data)
