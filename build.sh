#! /bin/sh

go build -o ./build/gates/and/and ./examples/gates/and/and.go
go build -o ./build/gates/or/or ./examples/gates/or/or.go
go build -o ./build/gates/xor/xor ./examples/gates/xor/xor.go
