async function main() {
    var axios = require("axios")
    
    var axiosInstance = axios.create();
    
    const resp = await axiosInstance.get("http://www.baidu.com")
    
    console.log(resp.data);
    
}

main()
