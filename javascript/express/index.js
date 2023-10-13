var express = require('express')
var app = express();

app.get('/', (req,res)=> {
    res.send('Hello World');
});


const port = 3000
app.listen(port,()=>{
    console.log("Server is running!");
})