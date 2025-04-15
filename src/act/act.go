package act

import "math"

type ActivationFunction int

const (
	Sigmoid ActivationFunction = iota
	Relu
	LeakyRelu
	Tanh
)

const (
	RELU_PARAM = 0.01
)

var actFuncNames = map[ActivationFunction]string{
	Sigmoid:   "Sigmoid",
	Relu:      "ReLU",
	LeakyRelu: "Leaky ReLU",
	Tanh:      "Tanh",
}

func (act ActivationFunction) Activate(x float64) float64 {
	switch act {
	case Sigmoid:
		return 1 / (1 + math.Exp(-x))
	case Relu:
		if x > 0 {
			return x
		}
		return 0
	case LeakyRelu:
		if x > 0 {
			return x
		}
		return x * RELU_PARAM
	case Tanh:
		return math.Tanh(x)
	default:
		panic("unknown activation function")
	}
}

func (act ActivationFunction) Derivative(a float64) float64 {
	switch act {
	case Sigmoid:
		return a * (1 - a)
	case Relu:
		if a > 0 {
			return 1
		}
		return 0
	case LeakyRelu:
		if a > 0 {
			return 1
		}
		return RELU_PARAM
	case Tanh:
		return 1 - a*a
	default:
		panic("unknown activation function")
	}
}

func (act ActivationFunction) String() string {
	return actFuncNames[act]
}
