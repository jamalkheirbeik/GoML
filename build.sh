#! /bin/sh

go build -o ./build/gates/and ./examples/gates/and/and.go
go build -o ./build/gates/or ./examples/gates/or/or.go
go build -o ./build/gates/xor ./examples/gates/xor/xor.go
