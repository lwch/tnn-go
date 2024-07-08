package net

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"google.golang.org/protobuf/proto"
)

type loadFunc func(name string, params []*tensor.Tensor, args map[string]float32) layer.Layer

var loadFuncs = map[string]loadFunc{
	"linear":     layer.LoadLinear,
	"dropout":    layer.LoadDropout,
	"conv1d":     layer.LoadConv1D,
	"conv2d":     layer.LoadConv2D,
	"maxpool1d":  layer.LoadMaxPool1D,
	"rnn":        layer.LoadRnn,
	"lstm":       layer.LoadLstm,
	"attention":  layer.LoadAttention,
	"attention1": layer.LoadAttention1,
	"layer_norm": layer.LoadLayerNorm,
	"rms_norm":   layer.LoadRMSNorm,
	"flatten":    layer.LoadFlatten,
	"embedding":  layer.LoadEmbedding,
	"rezero":     layer.LoadReZero,
	// activation
	"sigmoid": activation.LoadSigmoid,
	"tanh":    activation.LoadTanh,
	"relu":    activation.LoadRelu,
	"gelu":    activation.LoadGelu,
}

func RegisterLoadFunc(class string, fn loadFunc) {
	loadFuncs[class] = fn
}

type Net struct {
	layers    []layer.Layer
	device    consts.DeviceType
	optimizer optimizer.Optimizer
}

func New(device consts.DeviceType) *Net {
	return &Net{device: device}
}

func (n *Net) SetDevice(device consts.DeviceType) {
	n.device = device
}

func (n *Net) Add(layers ...layer.Layer) {
	n.layers = append(n.layers, layers...)
}

func (n *Net) SetOptimizer(optm optimizer.Optimizer) {
	n.optimizer = optm
}

func (n *Net) GetOptimizer() optimizer.Optimizer {
	return n.optimizer
}

func (n *Net) Params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	for _, l := range n.layers {
		ret = append(ret, l.Params()...)
	}
	return ret
}

func (n *Net) ParamCount() uint64 {
	var ret uint64
	for _, l := range n.layers {
		for _, p := range l.Params() {
			ret += uint64(p.ElemCount())
		}
	}
	return ret
}

func (n *Net) Save(dir string) error {
	f, err := os.Create(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = n.WriteTo(f)
	return err
}

func (n *Net) WriteTo(w io.Writer) (int64, error) {
	zw := zip.NewWriter(w)
	defer zw.Close()
	zw.RegisterCompressor(zip.Deflate, func(w io.Writer) (io.WriteCloser, error) {
		return zstd.NewWriter(w)
	})
	var net pb.Net
	net.Layers = make([]*pb.Layer, len(n.layers))
	params := make(map[string]*tensor.Tensor)
	for i := 0; i < len(n.layers); i++ {
		net.Layers[i] = new(pb.Layer)
		net.Layers[i].Class = n.layers[i].Class()
		net.Layers[i].Name = n.layers[i].Name()
		for j, p := range n.layers[i].Params() {
			var param pb.Param
			param.Type = uint32(p.ScalarType())
			param.ElemCount = p.ElemCount()
			param.Shapes = make([]int64, p.Dims())
			copy(param.Shapes, p.Shapes())
			param.File = fmt.Sprintf("layer_%d_param_%d.bin", i, j)
			params[param.File] = p
			net.Layers[i].Params = append(net.Layers[i].Params, &param)
		}
		net.Layers[i].Args = n.layers[i].Args()
	}
	if n.optimizer != nil {
		net.Optimizer = new(pb.Optimizer)
		net.Optimizer.Class = n.optimizer.GetName()
		var buf bytes.Buffer
		_, err := n.optimizer.GetOptions().WriteTo(&buf)
		if err != nil {
			return 0, err
		}
		net.Optimizer.Options = buf.Bytes()
		for i, arr := range n.optimizer.GetState() {
			var op pb.OptimizerParam
			for j, p := range arr {
				var param pb.Param
				param.Type = uint32(p.ScalarType())
				param.ElemCount = p.ElemCount()
				param.Shapes = make([]int64, p.Dims())
				copy(param.Shapes, p.Shapes())
				param.File = fmt.Sprintf("optimizer_%d_param_%d.bin", i, j)
				params[param.File] = p
				op.Params = append(op.Params, &param)
			}
			net.Optimizer.Params = append(net.Optimizer.Params, &op)
		}
	}
	data, err := proto.Marshal(&net)
	if err != nil {
		return 0, err
	}
	f, err := zw.CreateHeader(&zip.FileHeader{
		Name:     "SPEC",
		Method:   zip.Deflate,
		Modified: time.Now(),
	})
	if err != nil {
		return 0, err
	}
	cnt, err := io.Copy(f, bytes.NewReader(data))
	if err != nil {
		return 0, err
	}
	for file, param := range params {
		var bytes int64
		err = func() error {
			f, err := zw.CreateHeader(&zip.FileHeader{
				Name:     file,
				Method:   zip.Deflate,
				Modified: time.Now(),
			})
			if err != nil {
				return err
			}
			if !strings.HasPrefix(file, "optimizer_") {
				param = param.ToDevice(consts.KCPU)
			}
			switch param.ScalarType() {
			case consts.KUint8:
				bytes = 1
				return binary.Write(f, binary.BigEndian, param.Uint8Value())
			case consts.KInt8:
				bytes = 1
				return binary.Write(f, binary.BigEndian, param.Int8Value())
			case consts.KInt16:
				bytes = 2
				return binary.Write(f, binary.BigEndian, param.Int16Value())
			case consts.KInt32:
				bytes = 4
				return binary.Write(f, binary.BigEndian, param.Int32Value())
			case consts.KInt64:
				bytes = 8
				return binary.Write(f, binary.BigEndian, param.Int64Value())
			case consts.KHalf:
				bytes = 2
				return binary.Write(f, binary.BigEndian, param.HalfRaw())
			case consts.KFloat:
				bytes = 4
				return binary.Write(f, binary.BigEndian, param.Float32Value())
			case consts.KDouble:
				bytes = 8
				return binary.Write(f, binary.BigEndian, param.Float64Value())
			case consts.KBool:
				bytes = 1
				return binary.Write(f, binary.BigEndian, param.BoolValue())
			case consts.KBFloat16:
				bytes = 2
				return binary.Write(f, binary.BigEndian, param.BFloat16Raw())
			default:
				panic(fmt.Errorf("unsupported scalar type: %s", param.ScalarType().String()))
			}
		}()
		if err != nil {
			return 0, err
		}
		cnt += param.ElemCount() * bytes
	}
	return cnt, nil
}

func (n *Net) Load(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return err
	}
	_, err = n.ReadFrom(f, fi.Size())
	return err
}

func (n *Net) readSpec(r *zip.Reader) (*pb.Net, error) {
	f, err := r.Open("SPEC")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	var net pb.Net
	if err = proto.Unmarshal(data, &net); err != nil {
		return nil, err
	}
	return &net, nil
}

func buildParam[T uint8 | int8 | int16 | uint16 | int32 | int64 |
	float32 | float64 | bool](r io.Reader, cnt int64, shapes []int64, device consts.DeviceType,
	fn func(data []T, opts ...tensor.Option) *tensor.Tensor) (*tensor.Tensor, error) {
	data := make([]T, cnt)
	if err := binary.Read(r, binary.BigEndian, data); err != nil {
		return nil, err
	}
	t := fn(data,
		tensor.WithShapes(shapes...),
		tensor.WithDevice(device))
	return t, nil
}

func (n *Net) loadParam(r *zip.Reader, file string, t consts.ScalarType, cnt int64, shapes []int64) (*tensor.Tensor, error) {
	f, err := r.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	switch t {
	case consts.KUint8:
		return buildParam[uint8](f, cnt, shapes, n.device, tensor.FromUint8)
	case consts.KInt8:
		return buildParam[int8](f, cnt, shapes, n.device, tensor.FromInt8)
	case consts.KInt16:
		return buildParam[int16](f, cnt, shapes, n.device, tensor.FromInt16)
	case consts.KInt32:
		return buildParam[int32](f, cnt, shapes, n.device, tensor.FromInt32)
	case consts.KInt64:
		return buildParam[int64](f, cnt, shapes, n.device, tensor.FromInt64)
	case consts.KHalf:
		return buildParam[uint16](f, cnt, shapes, n.device, tensor.FromHalfRaw)
	case consts.KFloat:
		return buildParam[float32](f, cnt, shapes, n.device, tensor.FromFloat32)
	case consts.KDouble:
		return buildParam[float64](f, cnt, shapes, n.device, tensor.FromFloat64)
	case consts.KBool:
		return buildParam[bool](f, cnt, shapes, n.device, tensor.FromBool)
	case consts.KBFloat16:
		return buildParam[uint16](f, cnt, shapes, n.device, tensor.FromBFloat16Raw)
	default:
		panic(fmt.Errorf("unsupported scalar type: %s", t.String()))
	}
}

func (n *Net) ReadFrom(r io.ReaderAt, size int64) (int64, error) {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return 0, err
	}
	zr.RegisterDecompressor(zip.Deflate, func(r io.Reader) io.ReadCloser {
		zr, err := zstd.NewReader(r)
		runtime.Assert(err)
		return io.NopCloser(zr)
	})
	spec, err := n.readSpec(zr)
	if err != nil {
		return 0, err
	}
	layers := spec.GetLayers()
	n.layers = make([]layer.Layer, len(layers))
	var wg sync.WaitGroup
	wg.Add(len(layers))
	for i := 0; i < len(layers); i++ {
		class := layers[i].GetClass()
		fn := loadFuncs[class]
		if fn == nil {
			panic("unsupported " + class + " layer")
		}
		go func(i int) {
			defer wg.Done()
			var params []*tensor.Tensor
			for _, param := range layers[i].GetParams() {
				p, err := n.loadParam(zr,
					param.GetFile(),
					consts.ScalarType(param.GetType()),
					param.GetElemCount(),
					param.GetShapes())
				runtime.Assert(err)
				p.SetRequiresGrad(true)
				params = append(params, p)
			}
			n.layers[i] = fn(layers[i].GetName(), params, layers[i].GetArgs())
		}(i)
	}
	wg.Wait()

	if spec.GetOptimizer() != nil {
		switch spec.GetOptimizer().GetClass() {
		case "Adam":
			n.optimizer = optimizer.NewAdam(n.Params())
		case "AdamW":
			n.optimizer = optimizer.NewAdamW(n.Params())
		default:
			panic("unsupported optimizer: " + spec.GetOptimizer().GetClass())
		}
		_, err = n.optimizer.GetOptions().ReadFrom(bytes.NewReader(spec.GetOptimizer().GetOptions()))
		if err != nil {
			return 0, err
		}
		var state [][]*tensor.Tensor
		for _, params := range spec.GetOptimizer().GetParams() {
			var arr []*tensor.Tensor
			for _, param := range params.GetParams() {
				t, err := n.loadParam(zr,
					param.GetFile(),
					consts.ScalarType(param.GetType()),
					param.GetElemCount(),
					param.GetShapes())
				if err != nil {
					return 0, err
				}
				arr = append(arr, t)
			}
			state = append(state, arr)
		}
		n.optimizer.SetState(state)
	}
	return size, nil
}

func (n *Net) Layers() []layer.Layer {
	return n.layers
}

func (n *Net) ToScalarType(t consts.ScalarType) {
	for _, l := range n.layers {
		l.ToScalarType(t)
	}
}
