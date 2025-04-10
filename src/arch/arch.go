package arch

import "fmt"

type Arch struct {
	model []uint64
}

func NewArch(model ...uint64) Arch {
	if len(model) < 2 {
		panic("specify both input and output layers")
	}
	for i := range model {
		if model[i] == 0 {
			panic("cannot create a layer with 0 neurons")
		}
	}
	return Arch{model}
}

func (a *Arch) Print() {
	fmt.Println(a.model)
}

func (a *Arch) Size() int {
	return len(a.model)
}

func (a *Arch) NeuronsAt(i int) uint64 {
	return a.model[i]
}

// TODO: load architecture from file
// TODO: save architecture to file
