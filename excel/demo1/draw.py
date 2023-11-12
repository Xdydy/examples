import pygraphviz as pgv

# 创建一个新的无向图
graph = pgv.AGraph(strict=False, directed=False)

input_file = "output_relationship.txt"

# 渲染并保存图像
output_file = "undirected_graph.png"
with open(input_file,'r',encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(' ')
        op1 = parts[0]
        op2 = parts[1]
        graph.add_edge(op1,op2) 

    graph.draw(output_file, prog='dot', format='png')
          