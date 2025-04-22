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
	"time"
)

type gradient struct {
	Dbs []matrix.Matrix
	Dws []matrix.Matrix
}

func NewGradient(nn NeuralNetwork) gradient {
	size := nn.Arch.Size()
	g := gradient{
		Dbs: make([]matrix.Matrix, size),
		Dws: make([]matrix.Matrix, size),
	}

	g.Dbs[size-1] = *matrix.NewMatrix(nn.Activations[size-1].Rows, nn.Activations[size-1].Cols)
	g.Dws[size-1] = *matrix.NewMatrix(g.Dbs[size-1].Rows, nn.Activations[size-2].Rows)

	for j := nn.Arch.Size() - 2; j > 0; j-- {
		g.Dbs[j] = *matrix.NewMatrix(nn.Weights[j+1].Cols, g.Dbs[j+1].Cols)
		g.Dws[j] = *matrix.NewMatrix(g.Dbs[j].Rows, nn.Activations[j-1].Rows)
	}

	return g
}

func (g *gradient) addGradient(b gradient) {
	for i := range g.Dbs {
		g.Dbs[i] = *g.Dbs[i].AddMatrix(b.Dbs[i])
		g.Dws[i] = *g.Dws[i].AddMatrix(b.Dws[i])
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

	fanIn := arch.NeuronsAt(0)
	fanOut := arch.NeuronsAt(size - 1)
	for i := 1; i < size-1; i++ {
		w[i] = *matrix.NewMatrix(arch.NeuronsAt(i), arch.NeuronsAt(i-1))
		b[i] = *matrix.NewMatrix(arch.NeuronsAt(i), 1) // bias for each neuron
		a[i] = *matrix.NewMatrix(arch.NeuronsAt(i), 1)
		initWeights(w[i], hiddenAct, fanIn, fanOut)
	}

	i := size - 1
	w[i] = *matrix.NewMatrix(arch.NeuronsAt(i), arch.NeuronsAt(i-1))
	b[i] = *matrix.NewMatrix(arch.NeuronsAt(i), 1) // bias for each neuron
	a[i] = *matrix.NewMatrix(arch.NeuronsAt(i), 1)
	initWeights(w[i], outputAct, fanIn, fanOut)

	return NeuralNetwork{Arch: arch, Weights: w, Biases: b, Activations: a, HiddenActFunc: hiddenAct, OutputActFunc: outputAct}
}

func initWeights(ws matrix.Matrix, activation act.ActivationFunction, fanIn uint64, fanOut uint64) {
	switch activation {
	case act.Sigmoid:
		// Xavier uniform
		limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
		ws.Rand(-limit, limit)
	case act.Relu:
		// He uniform
		limit := math.Sqrt(6.0 / float64(fanIn))
		ws.Rand(-limit, limit)
	case act.LeakyRelu:
		// He uniform
		limit := math.Sqrt(6.0 / float64(fanIn))
		ws.Rand(-limit, limit)
	case act.Tanh:
		// Xavier uniform
		limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
		ws.Rand(-limit, limit)
	default:
		ws.Rand(-1, 1)
	}
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

func (nn *NeuralNetwork) Backprop(expected matrix.Matrix, predicted matrix.Matrix) gradient {
	g := NewGradient(*nn)
	g.Dbs[nn.Arch.Size()-1] = *predicted.SubMatrix(expected).ProdMatrix(*predicted.Apply(nn.OutputActFunc.Derivative))
	g.Dws[nn.Arch.Size()-1] = *g.Dbs[nn.Arch.Size()-1].DotMatrix(*nn.Activations[nn.Arch.Size()-2].Transpose())

	for j := nn.Arch.Size() - 2; j > 0; j-- {
		g.Dbs[j] = *nn.Weights[j+1].Transpose().DotMatrix(g.Dbs[j+1]).ProdMatrix(*nn.Activations[j].Apply(nn.HiddenActFunc.Derivative))
		g.Dws[j] = *g.Dbs[j].DotMatrix(*nn.Activations[j-1].Transpose())
	}

	return g
}

func (nn *NeuralNetwork) Train(dataset *dataset.Dataset, epochs int, rate float64, threshold float64, batchSize int) {
	for epoch := 1; epoch <= epochs; epoch++ {
		timeStart := time.Now()
		dataset.Shuffle()
		cost := 0.0
		batches := int(math.Ceil(float64(dataset.Input.Rows) / float64(batchSize)))

		for batch := range batches {
			start := batch * batchSize
			end := min(start+batchSize, int(dataset.Input.Rows))
			grads := NewGradient(*nn)

			for i := start; i < end; i++ {
				predicted := nn.Forward(dataset.Input.Data[i])
				expected := matrix.MatrixFrom1DArray(dataset.Output.Data[i]).Transpose()
				cost += nn.Cost(*expected, predicted)
				grads.addGradient(nn.Backprop(*expected, predicted))
			}

			nn.learn(grads, rate)
		}

		cost /= float64(dataset.Input.Rows)
		if cost < threshold {
			fmt.Printf("Target reached at Epoch: %d, Cost: %f\n", epoch, cost)
			break
		}
		fmt.Printf("Epoch = %d, Rate = %f, Cost = %f, Duration = %f\n", epoch, rate, cost, time.Since(timeStart).Seconds())
	}
}

func (nn *NeuralNetwork) learn(g gradient, rate float64) {
	for i := nn.Arch.Size() - 1; i > 0; i-- {
		nn.Weights[i] = *nn.Weights[i].SubMatrix(*g.Dws[i].Prod(rate))
		nn.Biases[i] = *nn.Biases[i].SubMatrix(*g.Dbs[i].Prod(rate))
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
