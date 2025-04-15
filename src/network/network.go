package network

import (
	"encoding/json"
	"fmt"
	"goml/src/act"
	"goml/src/arch"
	"goml/src/dataset"
	"goml/src/matrix"
	"math"
	"os"
)

type Gradient struct {
	Dbs []matrix.Matrix
	Dws []matrix.Matrix
}

func NewGradient(size int) Gradient {
	return Gradient{
		Dbs: make([]matrix.Matrix, size),
		Dws: make([]matrix.Matrix, size),
	}
}

type NeuralNetwork struct {
	Arch          arch.Arch              `json:"arch"`
	Weights       []matrix.Matrix        `json:"weights"`
	Biases        []matrix.Matrix        `json:"biases"`
	Activations   []matrix.Matrix        `json:"activations"`
	HiddenActFunc act.ActivationFunction `json:"hiddenActFunc"`
	OutputActFunc act.ActivationFunction `json:"outputActFunc"`
}

func NewNeuralNetwork(arch arch.Arch, hiddenAct act.ActivationFunction, outputAct act.ActivationFunction) NeuralNetwork {
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

	return NeuralNetwork{Arch: arch, Weights: w, Biases: b, Activations: a, HiddenActFunc: hiddenAct, OutputActFunc: outputAct}
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
	fmt.Printf("Hidden Layer Activation Function: %s\n", nn.HiddenActFunc.String())
	fmt.Printf("Output Layer Activation Function: %s\n", nn.OutputActFunc.String())
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

	for i := 1; i < nn.Arch.Size()-1; i++ {
		nn.Activations[i] = *nn.Weights[i].DotMatrix(nn.Activations[i-1]).AddMatrix(nn.Biases[i]).Apply(nn.HiddenActFunc.Activate)
	}

	i := nn.Arch.Size() - 1
	nn.Activations[i] = *nn.Weights[i].DotMatrix(nn.Activations[i-1]).AddMatrix(nn.Biases[i]).Apply(nn.OutputActFunc.Activate)

	return nn.Activations[nn.Arch.Size()-1]
}

func (nn *NeuralNetwork) Cost(expected matrix.Matrix, predicted matrix.Matrix) float64 {
	return 0.5 * expected.SubMatrix(predicted).Apply(func(x float64) float64 { return math.Pow(x, 2) }).Sum()
}

func (nn *NeuralNetwork) Backprop(expected matrix.Matrix, predicted matrix.Matrix) Gradient {
	size := nn.Arch.Size() - 1
	g := NewGradient(size)
	g.Dbs[size-1] = *predicted.SubMatrix(expected).ProdMatrix(*predicted.Apply(nn.OutputActFunc.Derivative))
	g.Dws[size-1] = *g.Dbs[size-1].DotMatrix(*nn.Activations[nn.Arch.Size()-2].Transpose())
	for i := size - 1; i > 0; i-- {
		g.Dbs[i-1] = *nn.Weights[i+1].Transpose().DotMatrix(g.Dbs[i]).ProdMatrix(*nn.Activations[i].Apply(nn.HiddenActFunc.Derivative))
		g.Dws[i-1] = *g.Dbs[i-1].DotMatrix(*nn.Activations[i-1].Transpose())
	}
	return g
}

func (nn *NeuralNetwork) Train(dataset *dataset.Dataset, epochs int, rate float64, threshold float64) {
	for epoch := 1; epoch <= epochs; epoch++ {
		cost := 0.0
		for i := range dataset.Input.Data {
			predicted := nn.Forward(dataset.Input.Data[i])
			expected := matrix.MatrixFrom1DArray(dataset.Output.Data[i]).Transpose()
			cost += nn.Cost(*expected, predicted)
			gradient := nn.Backprop(*expected, predicted)
			nn.Learn(gradient, rate)
		}

		cost /= float64(dataset.Input.Rows)
		if cost < threshold {
			fmt.Printf("Target reached at Epoch: %d, Cost: %f\n", epoch, cost)
			break
		}

		if epoch%1000 == 0 {
			fmt.Printf("Epoch = %d, Cost = %f\n", epoch, cost)
		}
		dataset.Shuffle()
	}
}

func (nn *NeuralNetwork) Learn(g Gradient, rate float64) {
	for i := nn.Arch.Size() - 1; i > 0; i-- {
		nn.Weights[i] = *nn.Weights[i].SubMatrix(*g.Dws[i-1].Prod(rate))
		nn.Biases[i] = *nn.Biases[i].SubMatrix(*g.Dbs[i-1].Prod(rate))
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
