const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

// 使用 body-parser 中间件来解析 JSON 请求体
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// 使用 body-parser 中间件来解析表单请求体

// 处理 POST 请求
app.post('/', (req, res) => {
  // 从请求体中获取 POST 数据
  const postData = req.body;

  // 在控制台中打印 POST 数据
  console.log('Received POST data:', postData);

  // 做其他处理...

  // 发送响应
  res.json(postData);
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
