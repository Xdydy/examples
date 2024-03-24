import {spawn} from 'child_process'

// Python代码，导入handler函数，调用它，并打印其返回值
const pythonCode = `
import json
from index import handler

event = {}  # 你可以在这里设置event的值
context = {}  # 你可以在这里设置context的值
result = handler(event, context)
print(json.dumps(result))
`

// 运行Python代码并获取handler函数的返回值
const output = await new Promise((resolve, reject) => {
    const python = spawn('python', ['-c', pythonCode])
    let data = ''
    python.stdout.on('data', (chunk) => {
        data += chunk
    })
    python.on('close', (code) => {
        if (code !== 0) {
            reject(`Process exited with code ${code}`)
        }
        resolve(data)
    })
})

// 解析Python脚本的输出
let result;
try {
    result = JSON.parse(output);
} catch (e) {
    console.error('Error parsing output:', e);
}

console.log(result)