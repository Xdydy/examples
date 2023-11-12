package main

import (
	"fmt"
)

// 定义接口类型 Callback
type Callback interface {
	Call(params ...interface{})
}

// 实现 Callback 接口的函数
type FuncWithoutArgs func()
type FuncWithArgs func(params ...interface{})

func (f FuncWithoutArgs) Call(params ...interface{}) {
	f()
}

func (f FuncWithArgs) Call(params ...interface{}) {
	f(params...)
}

func main() {
	// 创建一个字符串到 Callback 的映射
	callbackMap := make(map[string]Callback)

	// 添加无参数回调函数到映射
	callbackMap["funcWithoutArgs"] = FuncWithoutArgs(func() {
		fmt.Println("Function without arguments is called")
	})

	// 添加带参数回调函数到映射
	callbackMap["funcWithArgs"] = FuncWithArgs(func(params ...interface{}) {
		if len(params) == 2 {
			a, _ := params[0].(int)
			b, _ := params[1].(int)
			result := a + b
			fmt.Printf("Addition result with arguments: %d\n", result)
		} else {
			fmt.Println("Invalid number of arguments")
		}
	})

	// 调用映射中的回调函数
	callback, found := callbackMap["funcWithoutArgs"]
	if found {
		callback.Call()
	} else {
		fmt.Println("Callback not found")
	}

	callback, found = callbackMap["funcWithArgs"]
	if found {
		callback.Call(5, 3)
	} else {
		fmt.Println("Callback not found")
	}
}
