package dataset

import (
	"fmt"
	"goml/src/matrix"
)

type Dataset struct {
	input  matrix.Matrix
	output matrix.Matrix
}

func NewDataset(input matrix.Matrix, output matrix.Matrix) *Dataset {
	if input.Rows() != output.Rows() {
		panic("input and output array length mismatch")
	}
	return &Dataset{input: input, output: output}
}

func (ds *Dataset) Print() {
	fmt.Println("Input:")
	ds.input.Print()
	fmt.Println("Output:")
	ds.output.Print()
}

func (ds *Dataset) Input() *matrix.Matrix {
	return &ds.input
}

func (ds *Dataset) Output() *matrix.Matrix {
	return &ds.output
}

// TODO: read from files
// TODO: shuffle dataset
