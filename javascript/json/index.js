var jsonData = require("./test.json");

var mp = new Map();
for ( const key in jsonData ) {
    const value = jsonData[key]
    console.log(`${key}:${value}`)
    mp.set(key,value)
}

mp.forEach((value,key) => {
    console.log(`${key}:${value}`)
})