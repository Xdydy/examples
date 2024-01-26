package main

import (
	"fmt"
)

// 定义接口类型 Callback
type Callback interface {
	Call(params ...interface{}) interface{}
}

// 实现 Callback 接口的函数
type Function func(a,b int) int

func (fn Function) Call(params ...interface{}) interface{} {
	a, _ := params[0].(int)
	b, _ := params[1].(int)

	return fn(a,b)
}

func add(a,b int) int {
	return a+b
}

type Node struct {

}

func (n *Node) add(a,b int) int {
	return a+b
}

func main() {
	// 创建一个字符串到 Callback 的映射
	callbackMap := make(map[string]Callback)

	callbackMap["a"] = Function(add)
	
	var n Node
	callbackMap["b"] = Function(n.add)


	// 调用映射中的回调函数
	callback, found := callbackMap["a"]
	if found {
		result := callback.Call(1,2)
		fmt.Println(result)
	} else {
		fmt.Println("Callback not found")
	}

	callback, found = callbackMap["b"]
	if found {
		result := callback.Call(5, 3)
		fmt.Println(result)
	} else {
		fmt.Println("Callback not found")
	}
}
