package model

import (
	"fmt"
	"io/ioutil"

	"github.com/lwch/tnn/nn/loss"
	"github.com/lwch/tnn/nn/net"
	"github.com/lwch/tnn/nn/optimizer"
	"github.com/lwch/tnn/nn/params"
	"github.com/lwch/tnn/nn/pb"
	"gonum.org/v1/gonum/mat"
	"google.golang.org/protobuf/proto"
)

type Model struct {
	name       string
	trainCount uint64
	net        *net.Net
	loss       loss.Loss
	optimizer  optimizer.Optimizer
}

func New(net *net.Net, loss loss.Loss, optimizer optimizer.Optimizer) *Model {
	return &Model{
		name:      "<unset>",
		net:       net,
		loss:      loss,
		optimizer: optimizer,
	}
}

func (m *Model) SetName(name string) {
	m.name = name
}

func (m *Model) Predict(input mat.Matrix) mat.Matrix {
	return m.net.Forward(input)
}

func (m *Model) Train(input mat.Matrix, targets mat.Matrix) {
	pred := m.Predict(input)
	grad := m.loss.Grad(pred, targets)
	grads := m.net.Backward(grad)
	m.apply(grads)
	m.trainCount++
}

func (m *Model) Loss(input, targets mat.Matrix) float64 {
	pred := m.Predict(input)
	return m.loss.Loss(pred, targets)
}

func (m *Model) apply(grads []*params.Params) {
	params := m.net.Params()
	m.optimizer.Update(grads, params)
}

func (m *Model) ParamCount() uint64 {
	return m.net.ParamCount()
}

func (m *Model) Save(dir string) error {
	var model pb.Model
	model.Name = m.name
	model.TrainCount = m.trainCount
	model.ParamCount = m.ParamCount()
	model.Layers = m.net.SaveLayers()
	model.Loss = m.loss.Save()
	model.Optimizer = m.optimizer.Save()
	data, err := proto.Marshal(&model)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(dir, data, 0644)
}

func (m *Model) Load(dir string) error {
	data, err := ioutil.ReadFile(dir)
	if err != nil {
		return err
	}
	var model pb.Model
	err = proto.Unmarshal(data, &model)
	if err != nil {
		return err
	}
	m.name = model.GetName()
	m.trainCount = model.GetTrainCount()
	m.net = net.New()
	m.net.LoadLayers(model.GetLayers())
	m.loss = loss.Load(model.GetLoss())
	m.optimizer = optimizer.Load(model.GetOptimizer())
	return nil
}

func (m Model) Print() {
	fmt.Println("Model:", m.name)
	fmt.Println("Train Count:", m.trainCount)
	fmt.Println("Param Count:", m.ParamCount())
	loss.Print(m.loss)
	m.optimizer.Print()
	m.net.Print()
}
