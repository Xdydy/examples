import requests

# 下载一个zip文件
url = 'http://10.102.131.24:80/data/uploads/hello.zip'
# 不使用代理下载文件
r = requests.get(url, stream=True, proxies={'http': None, 'https': None})
print(r)
with open('code.zip', 'wb') as f:
    f.write(r.content)

# 解压文件
import zipfile
with zipfile.ZipFile('code.zip', 'r') as z:
    z.extractall('.')
    # 解压后的文件列表
    