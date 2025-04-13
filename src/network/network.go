package network

import (
	"encoding/json"
	"fmt"
	"goml/src/arch"
	"goml/src/dataset"
	"goml/src/matrix"
	"math"
	"os"
)

type NeuralNetwork struct {
	Arch        arch.Arch       `json:"arch"`
	Weights     []matrix.Matrix `json:"weights"`
	Biases      []matrix.Matrix `json:"biases"`
	Activations []matrix.Matrix `json:"activations"`
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

func NewNeuralNetwork(arch arch.Arch) NeuralNetwork {
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

	return NeuralNetwork{Arch: arch, Weights: w, Biases: b, Activations: a}
}

func (nn *NeuralNetwork) Print() {
	for i := range nn.Arch.Size() {
		fmt.Printf("Layer %d:\n", i)
		fmt.Println("Weights:")
		nn.Weights[i].Print()
		fmt.Println("Biases:")
		nn.Biases[i].Print()
		fmt.Println("Activations:")
		nn.Activations[i].Print()
	}
}

func (nn *NeuralNetwork) Randomize(min float64, max float64) {
	for i := range nn.Arch.Size() {
		nn.Weights[i].Rand(min, max)
		nn.Biases[i].Rand(min, max)
	}
}

func (nn *NeuralNetwork) Forward(input []float64) matrix.Matrix {
	inputMat := matrix.MatrixFrom1DArray(input)
	nn.Activations[0] = *inputMat.Transpose()

	for i := 1; i < nn.Arch.Size(); i++ {
		nn.Activations[i] = *nn.Weights[i].DotMatrix(nn.Activations[i-1]).AddMatrix(nn.Biases[i]).Apply(Sigmoid)
	}

	return nn.Activations[nn.Arch.Size()-1]
}

func (nn *NeuralNetwork) Cost(expected matrix.Matrix, predicted matrix.Matrix) float64 {
	return 0.5 * expected.SubMatrix(predicted).Apply(Square).Sum()
}

func (nn *NeuralNetwork) Backprop(expected matrix.Matrix, predicted matrix.Matrix, rate float64) {
	delta := predicted.SubMatrix(expected).ProdMatrix(*predicted.Apply(SigmoidDerivative))
	dw := delta.DotMatrix(*nn.Activations[nn.Arch.Size()-2].Transpose())
	nn.Weights[nn.Arch.Size()-1] = *nn.Weights[nn.Arch.Size()-1].SubMatrix(*dw.Prod(rate))
	nn.Biases[nn.Arch.Size()-1] = *nn.Biases[nn.Arch.Size()-1].SubMatrix(*delta.Prod(rate))

	for j := nn.Arch.Size() - 2; j > 0; j-- {
		delta = nn.Weights[j+1].Transpose().DotMatrix(*delta).ProdMatrix(*nn.Activations[j].Apply(SigmoidDerivative))
		dw := delta.DotMatrix(*nn.Activations[j-1].Transpose())
		nn.Weights[j] = *nn.Weights[j].SubMatrix(*dw.Prod(rate))
		nn.Biases[j] = *nn.Biases[j].SubMatrix(*delta.Prod(rate))
	}
}

func (nn *NeuralNetwork) Train(dataset *dataset.Dataset, epochs int, rate float64, threshold float64, maxRateReductions int, maxDecline int) {
	prevCost := 0.0
	reductions, decline := 0, 0

	for epoch := 1; epoch <= epochs; epoch++ {
		cost := 0.0
		for i := range dataset.Input.Data {
			predicted := nn.Forward(dataset.Input.Data[i])
			expected := matrix.MatrixFrom1DArray(dataset.Output.Data[i]).Transpose()
			cost += nn.Cost(*expected, predicted)
			nn.Backprop(*expected, predicted, rate)
		}

		cost /= float64(dataset.Input.Rows)
		if cost < threshold {
			fmt.Printf("Target reached at Epoch: %d, Cost: %f\n", epoch, cost)
			break
		}

		if cost > prevCost {
			decline++
		} else {
			decline = 0
		}
		prevCost = cost

		if decline > maxDecline {
			if reductions < maxRateReductions {
				rate *= 0.5
				reductions++
				decline = 0
				fmt.Printf("Reduced learning rate to %f\n", rate)
			} else {
				fmt.Printf("No improvements after %d reductions to learning rate. Epoch: %d, Cost: %f\n", maxRateReductions, epoch, cost)
				break
			}
		}

		if epoch%1000 == 0 {
			fmt.Printf("Epoch = %d, Cost = %f\n", epoch, cost)
		}
	}
}

func (nn *NeuralNetwork) Save() {
	b, err := json.Marshal(nn)
	if err != nil {
		panic(err)
	}
	file, err := os.Create("network.json")
	if err != nil {
		panic(err)
	}
	n, err := file.Write(b)
	_ = n
	if err != nil {
		panic(err)
	}
	fmt.Println("neural network saved successfuly to network.json")
}

func Load() (NeuralNetwork, error) {
	b, err := os.ReadFile("network.json")
	if err != nil {
		return NeuralNetwork{}, err
	}
	var nn NeuralNetwork
	if err := json.Unmarshal(b, &nn); err != nil {
		return NeuralNetwork{}, err
	}
	fmt.Println("neural network loaded successfuly from network.json")
	return nn, nil
}

// TODO: batching
// TODO: optimize memory usage and performance
// TODO: hardware acceleration
