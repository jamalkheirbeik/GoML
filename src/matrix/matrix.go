package matrix

import (
	"fmt"
	"math/rand"
)

type Matrix struct {
	Rows uint64      `json:"rows"`
	Cols uint64      `json:"cols"`
	Data [][]float64 `json:"data"`
}

func NewMatrix(rows uint64, cols uint64) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{rows, cols, data}
}

func (m *Matrix) Print() {
	for i := range m.Data {
		for j := range m.Data[i] {
			fmt.Printf("    %f", m.Data[i][j])
		}
		fmt.Println()
	}
	fmt.Println()
}

func (m *Matrix) Rand(min float64, max float64) {
	for i := range m.Data {
		for j := range m.Data[i] {
			m.Data[i][j] = min + rand.Float64()*(max-min)
		}
	}
}

func (m *Matrix) Add(x float64) *Matrix {
	n := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		for j := range m.Data[i] {
			n.Data[i][j] = m.Data[i][j] + x
		}
	}
	return n
}

func (m *Matrix) Sub(x float64) *Matrix {
	n := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		for j := range m.Data[i] {
			n.Data[i][j] = m.Data[i][j] - x
		}
	}
	return n
}

func (m *Matrix) Prod(x float64) *Matrix {
	n := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		for j := range m.Data[i] {
			n.Data[i][j] = m.Data[i][j] * x
		}
	}
	return n
}

func (m *Matrix) Copy() *Matrix {
	n := NewMatrix(m.Rows, m.Cols)
	for i := range n.Data {
		for j := range n.Data[i] {
			n.Data[i][j] = m.Data[i][j]
		}
	}
	return n
}

func (m *Matrix) AddMatrix(n Matrix) *Matrix {
	if m.Rows != n.Rows || m.Cols != n.Cols {
		panic(fmt.Sprintf("cannot perform \"A[%dx%d] + B[%dx%d]\": size mismatch\n", m.Rows, m.Cols, n.Rows, n.Cols))
	}
	r := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		for j := range m.Data[i] {
			r.Data[i][j] = m.Data[i][j] + n.Data[i][j]
		}
	}
	return r
}

func (m *Matrix) SubMatrix(n Matrix) *Matrix {
	if m.Rows != n.Rows || m.Cols != n.Cols {
		panic(fmt.Sprintf("cannot perform \"A[%dx%d] - B[%dx%d]\": size mismatch\n", m.Rows, m.Cols, n.Rows, n.Cols))
	}
	r := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		for j := range m.Data[i] {
			r.Data[i][j] = m.Data[i][j] - n.Data[i][j]
		}
	}
	return r
}

// hadamard multiplication (element-wise product)
func (m *Matrix) ProdMatrix(n Matrix) *Matrix {
	if m.Rows != n.Rows || m.Cols != n.Cols {
		panic(fmt.Sprintf("cannot perform \"A[%dx%d] * B[%dx%d]\": size mismatch\n", m.Rows, m.Cols, n.Rows, n.Cols))
	}
	r := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		for j := range m.Data[i] {
			r.Data[i][j] = m.Data[i][j] * n.Data[i][j]
		}
	}
	return r
}

func (m *Matrix) DotMatrix(n Matrix) *Matrix {
	if m.Cols != n.Rows {
		panic(fmt.Sprintf("invalid dot operation \"A[%dx%d] . B[%dx%d]\" COL_A != ROW_B\n", m.Rows, m.Cols, n.Rows, n.Cols))
	}
	out := NewMatrix(m.Rows, n.Cols)
	for i := range m.Rows {
		for j := range n.Cols {
			for k := range m.Cols {
				out.Data[i][j] += m.Data[i][k] * n.Data[k][j]
			}
		}
	}
	return out
}

func (m *Matrix) Transpose() *Matrix {
	n := NewMatrix(m.Cols, m.Rows)
	for i := range m.Data {
		for j := range m.Data[i] {
			n.Data[j][i] = m.Data[i][j]
		}
	}
	return n
}

func (m *Matrix) Fill(n float64) {
	for i := range m.Data {
		for j := range m.Data[i] {
			m.Data[i][j] = n
		}
	}
}

func (m *Matrix) Init(n [][]float64) {
	if int(m.Rows) != len(n) || int(m.Cols) != len(n[0]) {
		panic(fmt.Sprintf("cannot initialize matrix A[%dx%d] with array B[%dx%d]: size mismatch\n", m.Rows, m.Cols, len(n), len(n[0])))
	}
	for i := range m.Data {
		for j := range m.Data[i] {
			m.Data[i][j] = n[i][j]
		}
	}
}

func (m *Matrix) Trace() float64 {
	if m.Rows != m.Cols {
		panic(fmt.Sprintf("expected a square matrix [NxN] but got [%dx%d]\n", m.Rows, m.Cols))
	}
	sum := 0.0
	for i := range m.Data {
		sum += m.Data[i][i]
	}
	return sum
}

func (m *Matrix) Apply(modifier func(float64) float64) *Matrix {
	n := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		for j := range m.Data[i] {
			n.Data[i][j] = modifier(m.Data[i][j])
		}
	}
	return n
}

func (m *Matrix) Sum() float64 {
	sum := 0.0
	for i := range m.Data {
		for j := range m.Data[i] {
			sum += m.Data[i][j]
		}
	}
	return sum
}

func (m *Matrix) Assign(n Matrix) {
	if m.Rows != n.Rows || m.Cols != n.Cols {
		panic(fmt.Sprintf("cannot assign values of \"B[%dx%d] to A[%dx%d]\": size mismatch\n", n.Rows, n.Cols, m.Rows, m.Cols))
	}
	for i := range m.Data {
		for j := range m.Data[i] {
			m.Data[i][j] = n.Data[i][j]
		}
	}
}

func MatrixFrom1DArray(a []float64) *Matrix {
	m := NewMatrix(1, uint64(len(a)))
	for i := range m.Data[0] {
		m.Data[0][i] = a[i]
	}
	return m
}

func MatrixFrom2DArray(a [][]float64) *Matrix {
	m := NewMatrix(uint64(len(a)), uint64(len(a[0])))
	for i := range m.Data {
		for j := range m.Data[0] {
			m.Data[i][j] = a[i][j]
		}
	}
	return m
}

func NewFilledMatrix(value float64, rows uint64, cols uint64) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
		for j := range data[i] {
			data[i][j] = value
		}
	}
	return &Matrix{rows, cols, data}
}

// TODO: matrix power function (example: M^2)
// TODO: matrix determinant (for square matrices)
// TODO: matrix inversion (for square matrices)
