container_file = "output_container.txt"
device_file = "output_device.txt"
relationship_file = "output_connect.txt"

objects_name = []
for file_path in [container_file,device_file]:
    with open(file_path,'r',encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 0:
                table_name = str(parts[1])
                if table_name.endswith("基本信息表"):
                    objects_name.append(table_name[:-5])

relationship_list = []
with open(relationship_file,'r',encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) > 0:
            table_name = str(parts[1])
            if table_name.endswith("关系表"):
                table_name = table_name[:-3]
            elif table_name.endswith("关联关系"):
                table_name = table_name[:-4]
            
            relationship_list.append(table_name)

relationship = []
for obj1 in objects_name:
    for obj2 in objects_name:
        if obj1 == obj2:
            continue
        for relationship_str in relationship_list:
            if obj1 in relationship_str and obj2 in relationship_str:
                if (obj2,obj1) not in relationship:
                    relationship.append((obj1,obj2))

# print(objects_name)
# print('\n')
# print(relationship)

output_file = "output_relationship.txt"
with open(output_file,'w',encoding='utf-8') as file:
    for _tuple in relationship:
        for _ in _tuple: 
            file.write(_ + " ")   
        file.write('\n')