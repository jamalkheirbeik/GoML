package matrix

import (
	"fmt"
	"math/rand"
)

type Matrix struct {
	rows uint64
	cols uint64
	data [][]float64
}

func NewMatrix(rows uint64, cols uint64) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{rows, cols, data}
}

func (m *Matrix) Rows() uint64 {
	return m.rows
}

func (m *Matrix) Cols() uint64 {
	return m.cols
}

func (m *Matrix) Data() [][]float64 {
	return m.data
}

func (m *Matrix) Print() {
	for i := range m.data {
		for j := range m.data[i] {
			fmt.Printf("    %f", m.data[i][j])
		}
		fmt.Println()
	}
	fmt.Println()
}

func (m *Matrix) Rand(min float64, max float64) {
	for i := range m.data {
		for j := range m.data[i] {
			m.data[i][j] = min + rand.Float64()*(max-min)
		}
	}
}

func (m *Matrix) Add(x float64) *Matrix {
	n := NewMatrix(m.rows, m.cols)
	for i := range m.data {
		for j := range m.data[i] {
			n.data[i][j] = m.data[i][j] + x
		}
	}
	return n
}

func (m *Matrix) Sub(x float64) *Matrix {
	n := NewMatrix(m.rows, m.cols)
	for i := range m.data {
		for j := range m.data[i] {
			n.data[i][j] = m.data[i][j] - x
		}
	}
	return n
}

func (m *Matrix) Prod(x float64) *Matrix {
	n := NewMatrix(m.rows, m.cols)
	for i := range m.data {
		for j := range m.data[i] {
			n.data[i][j] = m.data[i][j] * x
		}
	}
	return n
}

func (m *Matrix) Copy() *Matrix {
	n := NewMatrix(m.rows, m.cols)
	for i := range n.data {
		for j := range n.data[i] {
			n.data[i][j] = m.data[i][j]
		}
	}
	return n
}

func (m *Matrix) AddMatrix(n Matrix) *Matrix {
	if m.rows != n.rows || m.cols != n.cols {
		panic(fmt.Sprintf("cannot perform \"A[%dx%d] + B[%dx%d]\": size mismatch\n", m.rows, m.cols, n.rows, n.cols))
	}
	r := NewMatrix(m.rows, m.cols)
	for i := range m.data {
		for j := range m.data[i] {
			r.data[i][j] = m.data[i][j] + n.data[i][j]
		}
	}
	return r
}

func (m *Matrix) SubMatrix(n Matrix) *Matrix {
	if m.rows != n.rows || m.cols != n.cols {
		panic(fmt.Sprintf("cannot perform \"A[%dx%d] - B[%dx%d]\": size mismatch\n", m.rows, m.cols, n.rows, n.cols))
	}
	r := NewMatrix(m.rows, m.cols)
	for i := range m.data {
		for j := range m.data[i] {
			r.data[i][j] = m.data[i][j] - n.data[i][j]
		}
	}
	return r
}

// hadamard multiplication (element-wise product)
func (m *Matrix) ProdMatrix(n Matrix) *Matrix {
	if m.rows != n.rows || m.cols != n.cols {
		panic(fmt.Sprintf("cannot perform \"A[%dx%d] * B[%dx%d]\": size mismatch\n", m.rows, m.cols, n.rows, n.cols))
	}
	r := NewMatrix(m.rows, m.cols)
	for i := range m.data {
		for j := range m.data[i] {
			r.data[i][j] = m.data[i][j] * n.data[i][j]
		}
	}
	return r
}

func (m *Matrix) DotMatrix(n Matrix) *Matrix {
	if m.cols != n.rows {
		panic(fmt.Sprintf("invalid dot operation \"A[%dx%d] . B[%dx%d]\" COL_A != ROW_B\n", m.rows, m.cols, n.rows, n.cols))
	}
	out := NewMatrix(m.rows, n.cols)
	for i := range m.rows {
		for j := range n.cols {
			for k := range m.cols {
				out.data[i][j] += m.data[i][k] * n.data[k][j]
			}
		}
	}
	return out
}

func (m *Matrix) Transpose() *Matrix {
	n := NewMatrix(m.cols, m.rows)
	for i := range m.data {
		for j := range m.data[i] {
			n.data[j][i] = m.data[i][j]
		}
	}
	return n
}

func (m *Matrix) Fill(n float64) {
	for i := range m.data {
		for j := range m.data[i] {
			m.data[i][j] = n
		}
	}
}

func (m *Matrix) Init(n [][]float64) {
	if int(m.rows) != len(n) || int(m.cols) != len(n[0]) {
		panic(fmt.Sprintf("cannot initialize matrix A[%dx%d] with array B[%dx%d]: size mismatch\n", m.rows, m.cols, len(n), len(n[0])))
	}
	for i := range m.data {
		for j := range m.data[i] {
			m.data[i][j] = n[i][j]
		}
	}
}

func (m *Matrix) Trace() float64 {
	if m.rows != m.cols {
		panic(fmt.Sprintf("expected a square matrix [NxN] but got [%dx%d]\n", m.rows, m.cols))
	}
	sum := 0.0
	for i := range m.data {
		sum += m.data[i][i]
	}
	return sum
}

func (m *Matrix) Apply(modifier func(float64) float64) *Matrix {
	n := NewMatrix(m.rows, m.cols)
	for i := range m.data {
		for j := range m.data[i] {
			n.data[i][j] = modifier(m.data[i][j])
		}
	}
	return n
}

func (m *Matrix) Sum() float64 {
	sum := 0.0
	for i := range m.data {
		for j := range m.data[i] {
			sum += m.data[i][j]
		}
	}
	return sum
}

func (m *Matrix) Get(i int, j int) float64 {
	return m.data[i][j]
}

func (m *Matrix) GetRow(i int) []float64 {
	return m.data[i]
}

func (m *Matrix) Assign(n Matrix) {
	if m.rows != n.rows || m.cols != n.cols {
		panic(fmt.Sprintf("cannot assign values of \"B[%dx%d] to A[%dx%d]\": size mismatch\n", n.rows, n.cols, m.rows, m.cols))
	}
	for i := range m.data {
		for j := range m.data[i] {
			m.data[i][j] = n.data[i][j]
		}
	}
}

func MatrixFrom1DArray(a []float64) *Matrix {
	m := NewMatrix(1, uint64(len(a)))
	for i := range m.data[0] {
		m.data[0][i] = a[i]
	}
	return m
}

func MatrixFrom2DArray(a [][]float64) *Matrix {
	m := NewMatrix(uint64(len(a)), uint64(len(a[0])))
	for i := range m.data {
		for j := range m.data[0] {
			m.data[i][j] = a[i][j]
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

// TODO: maatrix power function (example: M^2)
// TODO: matrix determinant (for square matrices)
// TODO: matrix inversion (for square matrices)
