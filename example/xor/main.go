package main

import (
	"fmt"
	"os"
	"time"
	"tnn/internal/initializer"
	"tnn/internal/nn/layer"
	"tnn/internal/nn/layer/activation"
	"tnn/internal/nn/loss"
	"tnn/internal/nn/model"
	"tnn/internal/nn/net"
	"tnn/internal/nn/optimizer"

	"github.com/lwch/runtime"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 0.0001
const modelFile = "xor.model"

var input = mat.NewDense(4, 2, []float64{
	0, 0,
	0, 1,
	1, 0,
	1, 1,
})

var output = mat.NewDense(4, 1, []float64{
	0,
	1,
	1,
	0,
})

func main() {
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		train()
		return
	}
	predict()
}

func train() {
	const hidden = 16

	initializer := initializer.NewNormal(1, 0.5)

	var net net.Net
	net.Set(
		layer.NewDense(hidden, initializer),
		activation.NewSigmoid(),
		layer.NewDense(1, initializer),
	)
	loss := loss.NewMSE()
	// optimizer := optimizer.NewSGD(lr, 0)
	optimizer := optimizer.NewAdam(lr, 0, 0.9, 0.999, 1e-8)
	m := model.New(&net, loss, optimizer)

	p := plot.New()
	p.Title.Text = "xor train model"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"

	var points plotter.XYs
	begin := time.Now()
	var i int
	for {
		m.Train(input, output)
		if i%100 == 0 {
			loss := m.Loss(input, output)
			fmt.Printf("Epoch: %d, Loss: %.05f\n", i, loss)
			points = append(points, plotter.XY{X: float64(i), Y: loss})
			if loss < 1e-10 {
				break
			}
		}
		i++
	}
	fmt.Printf("train cost: %s\n", time.Since(begin).String())
	fmt.Println("predict:")
	pred := m.Predict(input)
	for i := 0; i < 4; i++ {
		fmt.Printf("%d xor %d: %.2f\n",
			int(input.At(i, 0)), int(input.At(i, 1)),
			pred.At(i, 0))
	}

	l, err := plotter.NewLine(points)
	runtime.Assert(err)
	l.LineStyle.Color = plotutil.DarkColors[0]
	p.Add(l)
	p.Save(8*vg.Inch, 8*vg.Inch, "xor.png")

	runtime.Assert(m.Save(modelFile))
}

func predict() {
	var m model.Model
	runtime.Assert(m.Load(modelFile))
	pred := m.Predict(input)
	for i := 0; i < 4; i++ {
		fmt.Printf("%d xor %d: %.2f\n",
			int(input.At(i, 0)), int(input.At(i, 1)),
			pred.At(i, 0))
	}
}
