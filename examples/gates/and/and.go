package main

import (
	"fmt"
	"goml/src/arch"
	"goml/src/dataset"
	"goml/src/matrix"
	"goml/src/network"
)

const (
	LEARNING_RATE = 0.1
	EPOCHS        = 10_000
	THRESHOLD     = 0.01
)

var (
	input = [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	output = [][]float64{
		{0},
		{0},
		{0},
		{1},
	}
)

func main() {
	dataset := dataset.NewDataset(input, output)
	nn := network.NewNeuralNetwork(arch.NewArch(2, 2, 1))
	nn.Randomize(-1, 1)

	fmt.Printf("\nBefore Training:\n\n")
	for i, row := range input {
		predicted := nn.Forward(row)
		expected := matrix.MatrixFrom1DArray(output[i]).Transpose()
		fmt.Printf("Input: %v, Output: %v\n", row, output[i])
		fmt.Printf("Predicted: %v, Cost: %f\n", predicted.Data, nn.Cost(*expected, predicted))
	}

	nn.Train(dataset, EPOCHS, LEARNING_RATE, THRESHOLD)

	fmt.Printf("\nAfter Training:\n\n")
	for i, row := range input {
		predicted := nn.Forward(row)
		expected := matrix.MatrixFrom1DArray(output[i]).Transpose()
		fmt.Printf("Input: %v, Output: %v\n", row, output[i])
		fmt.Printf("Predicted: %v, Cost: %f\n", predicted.Data, nn.Cost(*expected, predicted))
	}
}
