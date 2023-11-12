async function main() {
    const url = "http://www.baidu.com"
    const response = await fetch(url, {
        method: 'GET'
    })
    const data = await response.text();
    console.log(data)
}

main()