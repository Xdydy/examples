const express = require('express');
const app = express();
const port = 3000;

// 引入另一个模块
const calculateFunction = require('./tmp');

// 定义一个路由
app.get('/', (req, res) => {
  // 调用另一个模块的函数
  const result = calculateFunction.handlers({"hello":"world"},null);
  res.json(result);
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
