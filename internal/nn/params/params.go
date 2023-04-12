package params

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Params map[string]*mat.Dense

func New() *Params {
	return &Params{}
}

func (params *Params) Copy(ps Params) {
	*params = make(Params)
	for name, value := range ps {
		var dense mat.Dense
		dense.CloneFrom(value)
		(*params)[name] = &dense
	}
}

func (params Params) Add(grads *Params) {
	for name, grad := range *grads {
		p := params[name]
		if p == nil {
			continue
		}
		p.Add(p, grad)
	}
}

func (params *Params) Apply(fn func(i, j int, v float64) float64) {
	for _, grad := range *params {
		grad.Apply(fn, grad)
	}
}

func (params Params) Range(fn func(name string, dense *mat.Dense)) {
	for name, dense := range params {
		fn(name, dense)
	}
}

func (params *Params) Init(name string, rows, cols int) {
	(*params)[name] = mat.NewDense(rows, cols, nil)
}

func (params Params) Get(name string) *mat.Dense {
	return params[name]
}

func (params Params) Print() {
	if len(params) == 0 {
		return
	}
	fmt.Println("============ params ==================")
	for name, dense := range params {
		fmt.Println(name)
		fmt.Println(mat.Formatted(dense))
	}
	fmt.Println("======================================")
}

func (params Params) Size() int {
	return len(params)
}
