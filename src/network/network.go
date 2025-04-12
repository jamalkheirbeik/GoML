package network

import (
	"fmt"
	"goml/src/arch"
	"goml/src/dataset"
	"goml/src/matrix"
	"math"
)

type NerualNetwork struct {
	arch        arch.Arch
	weights     []matrix.Matrix
	biases      []matrix.Matrix
	activations []matrix.Matrix
}

func Square(x float64) float64 {
	return math.Pow(x, 2)
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func NewNeuralNetwork(arch arch.Arch) NerualNetwork {
	size := arch.Size()
	w := make([]matrix.Matrix, size)
	b := make([]matrix.Matrix, size)
	a := make([]matrix.Matrix, size)
	// init input layer
	w[0] = *matrix.NewMatrix(0, 0)
	b[0] = *matrix.NewMatrix(0, 0)
	a[0] = *matrix.NewMatrix(arch.NeuronsAt(0), 1)

	for i := 1; i < size; i++ {
		w[i] = *matrix.NewMatrix(arch.NeuronsAt(i), arch.NeuronsAt(i-1))
		b[i] = *matrix.NewMatrix(arch.NeuronsAt(i), 1) // bias for each neuron
		a[i] = *matrix.NewMatrix(arch.NeuronsAt(i), 1)
	}

	return NerualNetwork{arch: arch, weights: w, biases: b, activations: a}
}

func (nn *NerualNetwork) Print() {
	for i := range nn.arch.Size() {
		fmt.Printf("Layer %d:\n", i)
		fmt.Println("Weights:")
		nn.weights[i].Print()
		fmt.Println("Biases:")
		nn.biases[i].Print()
		fmt.Println("Activations:")
		nn.activations[i].Print()
	}
}

func (nn *NerualNetwork) Randomize(min float64, max float64) {
	for i := range nn.arch.Size() {
		nn.weights[i].Rand(min, max)
		nn.biases[i].Rand(min, max)
	}
}

func (nn *NerualNetwork) Forward(input []float64) matrix.Matrix {
	inputMat := matrix.MatrixFrom1DArray(input)
	nn.activations[0] = *inputMat.Transpose()

	for i := 1; i < nn.arch.Size(); i++ {
		nn.activations[i] = *nn.weights[i].DotMatrix(nn.activations[i-1]).AddMatrix(nn.biases[i]).Apply(Sigmoid)
	}

	return nn.activations[nn.arch.Size()-1]
}

func (nn *NerualNetwork) Cost(expected matrix.Matrix, predicted matrix.Matrix) float64 {
	return 0.5 * expected.SubMatrix(predicted).Apply(Square).Sum()
}

func (nn *NerualNetwork) Backprop(expected matrix.Matrix, predicted matrix.Matrix, rate float64) {
	delta := predicted.SubMatrix(expected).ProdMatrix(*predicted.Apply(SigmoidDerivative))
	dw := delta.DotMatrix(*nn.activations[nn.arch.Size()-2].Transpose())
	nn.weights[nn.arch.Size()-1] = *nn.weights[nn.arch.Size()-1].SubMatrix(*dw.Prod(rate))
	nn.biases[nn.arch.Size()-1] = *nn.biases[nn.arch.Size()-1].SubMatrix(*delta.Prod(rate))

	for j := nn.arch.Size() - 2; j > 0; j-- {
		delta = nn.weights[j+1].Transpose().DotMatrix(*delta).ProdMatrix(*nn.activations[j].Apply(SigmoidDerivative))
		dw := delta.DotMatrix(*nn.activations[j-1].Transpose())
		nn.weights[j] = *nn.weights[j].SubMatrix(*dw.Prod(rate))
		nn.biases[j] = *nn.biases[j].SubMatrix(*delta.Prod(rate))
	}
}

func (nn *NerualNetwork) Train(dataset *dataset.Dataset, epochs int, rate float64, threshold float64) {
	for epoch := 1; epoch <= epochs*100; epoch++ {
		totalCost := 0.0
		for i := range dataset.Input().Data() {
			predicted := nn.Forward(dataset.Input().GetRow(i))
			expected := matrix.MatrixFrom1DArray(dataset.Output().GetRow(i))
			totalCost += nn.Cost(*expected, predicted)
			nn.Backprop(*expected, predicted, rate)
		}

		if totalCost < threshold {
			fmt.Printf("Early training exit. Epoch: %d, Total Cost: %f\n", epoch, totalCost)
			break
		}

		if epoch%1000 == 0 {
			fmt.Printf("Epoch = %d, Cost = %f\n", epoch, totalCost/float64(dataset.Input().Rows()))
		}
	}
}

// TODO: train until (cost <= threshold) or we hit max epochs
// TODO: load/save neural network from/to file
// TODO: batching
// TODO: optimize memory usage and performance
// TODO: hardware acceleration
