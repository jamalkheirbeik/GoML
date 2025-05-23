package dataset

import (
	"encoding/json"
	"fmt"
	"goml/src/matrix"
	"math/rand"
	"os"
)

type Dataset struct {
	Input  matrix.Matrix `json:"input"`
	Output matrix.Matrix `json:"output"`
}

func NewDataset(input [][]float64, output [][]float64) *Dataset {
	if len(input) != len(output) {
		panic("input and output array length mismatch")
	}
	return &Dataset{Input: *matrix.MatrixFrom2DArray(input), Output: *matrix.MatrixFrom2DArray(output)}
}

func (ds *Dataset) Print() {
	fmt.Println("Input:")
	ds.Input.Print()
	fmt.Println("Output:")
	ds.Output.Print()
}

func (ds *Dataset) Save() {
	b, err := json.Marshal(ds)
	if err != nil {
		panic(err)
	}
	file, err := os.Create("dataset.json")
	if err != nil {
		panic(err)
	}
	n, err := file.Write(b)
	_ = n
	if err != nil {
		panic(err)
	}
	fmt.Println("dataset saved successfuly to dataset.json")
}

func Load() (Dataset, error) {
	b, err := os.ReadFile("dataset.json")
	if err != nil {
		return Dataset{}, err
	}
	var ds Dataset
	if err := json.Unmarshal(b, &ds); err != nil {
		return Dataset{}, err
	}
	fmt.Println("dataset loaded successfuly from dataset.json")
	return ds, nil
}

func (ds *Dataset) Shuffle() {
	rand.Shuffle(len(ds.Input.Data), func(i, j int) {
		ds.Input.Data[i], ds.Input.Data[j] = ds.Input.Data[j], ds.Input.Data[i]
		ds.Output.Data[i], ds.Output.Data[j] = ds.Output.Data[j], ds.Output.Data[i]
	})
}

// TODO: normalization functions
