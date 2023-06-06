package main

import (
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"gorgonia.org/gorgonia"
)

type model struct {
	g          *gorgonia.ExprGraph
	vm         gorgonia.VM
	pred, loss *gorgonia.Node
	net        *net.Net
	params     gorgonia.Nodes
}

func newModel() *model {
	return &model{
		g: gorgonia.NewGraph(),
	}
}

func (m *model) Close() {
	if m.vm != nil {
		m.vm.Close()
	}
}

func (m *model) Compile(net *net.Net, loss loss.Loss, x, y *gorgonia.Node) {
	m.net = net
	input := x
	for _, l := range net.Layers() {
		switch ln := l.(type) {
		case *layer.Dense:
			input = ln.Forward(input)
		case *activation.ReLU:
			input = ln.Forward(input)
		}
	}
	m.params = m.Params()
	m.pred = input
	m.loss = loss.Loss(y, m.pred)
	_, err := gorgonia.Grad(m.loss, m.params...)
	runtime.Assert(err)
	m.vm = gorgonia.NewTapeMachine(m.g, gorgonia.BindDualValues(m.params...))
}

func (m *model) Train(optimizer optimizer.Optimizer) gorgonia.Value {
	if m.vm == nil {
		panic("model not compiled")
	}
	m.vm.Reset()
	runtime.Assert(m.vm.RunAll())
	runtime.Assert(optimizer.Step(m.params))
	return m.loss.Value()
}

func (m *model) Evaluate() (gorgonia.Value, gorgonia.Value) {
	runtime.Assert(m.vm.RunAll())
	return m.pred.Value(), m.loss.Value()
}

func (m *model) G() *gorgonia.ExprGraph {
	return m.g
}

func (m *model) Params() gorgonia.Nodes {
	var params gorgonia.Nodes
	for _, list := range m.net.Params() {
		for name, p := range list {
			params = append(params, gorgonia.NodeFromAny(m.g, p, gorgonia.WithName(name)))
		}
	}
	return params
}
