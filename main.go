package main

import (
	"os"

	"github.com/user/ollama-dl/cmd"
)

func main() {
	if err := cmd.Root().Execute(); err != nil {
		os.Exit(1)
	}
}
