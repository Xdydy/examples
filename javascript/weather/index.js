const params = new URLSearchParams({
    key: "a95c1331bbb60f7ea885b4acef3442ce",
    city: "440100",
    extensions: "base",
    output: "JSON"
});

fetch(`https://restapi.amap.com/v3/weather/weatherInfo?${params.toString()}`, {
    method: "GET",
    headers: {
        'Content-Type': 'application/json'
    }
})
.then(response => response.json())
.then(data => console.log(data.lives[0]))