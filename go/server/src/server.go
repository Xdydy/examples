package main

import (
	"fmt"
	"net"
	"os"
)

func main() {
	listen, err := net.Listen("tcp", "localhost:8080")
	if err != nil {
		fmt.Println("Error listening:", err)
		os.Exit(1)
	}
	defer listen.Close()

	fmt.Println("Server is listening on localhost:8080")

	for {
		conn, err := listen.Accept()
		if err != nil {
			fmt.Println("Error accepting connection:", err)
			os.Exit(1)
		}
		go handleConnection(conn)
	}
}

func handleConnection(conn net.Conn) {
	buffer := make([]byte, 1024)
	_, err := conn.Read(buffer)
	if err != nil {
		fmt.Println("Error reading:", err)
		conn.Close()
		return
	}

	receivedMessage := string(buffer)
	fmt.Println("Received message: ", receivedMessage)

	responseMessage := "Hello, client! I received your message."
	_, err = conn.Write([]byte(responseMessage))
	if err != nil {
		fmt.Println("Error writing:", err)
	}

	conn.Close()
}
