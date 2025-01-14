// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.30.0
// 	protoc        v4.22.5
// source: model.proto

package pb

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type Param struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Type      uint32  `protobuf:"varint,1,opt,name=type,proto3" json:"type,omitempty"`
	ElemCount int64   `protobuf:"varint,2,opt,name=elem_count,json=elemCount,proto3" json:"elem_count,omitempty"`
	Name      string  `protobuf:"bytes,3,opt,name=name,proto3" json:"name,omitempty"`
	Shapes    []int64 `protobuf:"varint,4,rep,packed,name=shapes,proto3" json:"shapes,omitempty"`
	File      string  `protobuf:"bytes,5,opt,name=file,proto3" json:"file,omitempty"`
}

func (x *Param) Reset() {
	*x = Param{}
	if protoimpl.UnsafeEnabled {
		mi := &file_model_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Param) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Param) ProtoMessage() {}

func (x *Param) ProtoReflect() protoreflect.Message {
	mi := &file_model_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Param.ProtoReflect.Descriptor instead.
func (*Param) Descriptor() ([]byte, []int) {
	return file_model_proto_rawDescGZIP(), []int{0}
}

func (x *Param) GetType() uint32 {
	if x != nil {
		return x.Type
	}
	return 0
}

func (x *Param) GetElemCount() int64 {
	if x != nil {
		return x.ElemCount
	}
	return 0
}

func (x *Param) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *Param) GetShapes() []int64 {
	if x != nil {
		return x.Shapes
	}
	return nil
}

func (x *Param) GetFile() string {
	if x != nil {
		return x.File
	}
	return ""
}

type Layer struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Class  string             `protobuf:"bytes,1,opt,name=class,proto3" json:"class,omitempty"`
	Name   string             `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
	Params []*Param           `protobuf:"bytes,3,rep,name=params,proto3" json:"params,omitempty"`
	Args   map[string]float32 `protobuf:"bytes,4,rep,name=args,proto3" json:"args,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"fixed32,2,opt,name=value,proto3"`
}

func (x *Layer) Reset() {
	*x = Layer{}
	if protoimpl.UnsafeEnabled {
		mi := &file_model_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Layer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Layer) ProtoMessage() {}

func (x *Layer) ProtoReflect() protoreflect.Message {
	mi := &file_model_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Layer.ProtoReflect.Descriptor instead.
func (*Layer) Descriptor() ([]byte, []int) {
	return file_model_proto_rawDescGZIP(), []int{1}
}

func (x *Layer) GetClass() string {
	if x != nil {
		return x.Class
	}
	return ""
}

func (x *Layer) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *Layer) GetParams() []*Param {
	if x != nil {
		return x.Params
	}
	return nil
}

func (x *Layer) GetArgs() map[string]float32 {
	if x != nil {
		return x.Args
	}
	return nil
}

type OptimizerParam struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Params []*Param `protobuf:"bytes,1,rep,name=params,proto3" json:"params,omitempty"`
}

func (x *OptimizerParam) Reset() {
	*x = OptimizerParam{}
	if protoimpl.UnsafeEnabled {
		mi := &file_model_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *OptimizerParam) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*OptimizerParam) ProtoMessage() {}

func (x *OptimizerParam) ProtoReflect() protoreflect.Message {
	mi := &file_model_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use OptimizerParam.ProtoReflect.Descriptor instead.
func (*OptimizerParam) Descriptor() ([]byte, []int) {
	return file_model_proto_rawDescGZIP(), []int{2}
}

func (x *OptimizerParam) GetParams() []*Param {
	if x != nil {
		return x.Params
	}
	return nil
}

type Optimizer struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Class   string            `protobuf:"bytes,1,opt,name=class,proto3" json:"class,omitempty"`
	Options []byte            `protobuf:"bytes,2,opt,name=options,proto3" json:"options,omitempty"`
	Params  []*OptimizerParam `protobuf:"bytes,3,rep,name=params,proto3" json:"params,omitempty"`
}

func (x *Optimizer) Reset() {
	*x = Optimizer{}
	if protoimpl.UnsafeEnabled {
		mi := &file_model_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Optimizer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Optimizer) ProtoMessage() {}

func (x *Optimizer) ProtoReflect() protoreflect.Message {
	mi := &file_model_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Optimizer.ProtoReflect.Descriptor instead.
func (*Optimizer) Descriptor() ([]byte, []int) {
	return file_model_proto_rawDescGZIP(), []int{3}
}

func (x *Optimizer) GetClass() string {
	if x != nil {
		return x.Class
	}
	return ""
}

func (x *Optimizer) GetOptions() []byte {
	if x != nil {
		return x.Options
	}
	return nil
}

func (x *Optimizer) GetParams() []*OptimizerParam {
	if x != nil {
		return x.Params
	}
	return nil
}

type Net struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Layers    []*Layer   `protobuf:"bytes,1,rep,name=layers,proto3" json:"layers,omitempty"`
	Optimizer *Optimizer `protobuf:"bytes,2,opt,name=optimizer,proto3" json:"optimizer,omitempty"`
}

func (x *Net) Reset() {
	*x = Net{}
	if protoimpl.UnsafeEnabled {
		mi := &file_model_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Net) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Net) ProtoMessage() {}

func (x *Net) ProtoReflect() protoreflect.Message {
	mi := &file_model_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Net.ProtoReflect.Descriptor instead.
func (*Net) Descriptor() ([]byte, []int) {
	return file_model_proto_rawDescGZIP(), []int{4}
}

func (x *Net) GetLayers() []*Layer {
	if x != nil {
		return x.Layers
	}
	return nil
}

func (x *Net) GetOptimizer() *Optimizer {
	if x != nil {
		return x.Optimizer
	}
	return nil
}

var File_model_proto protoreflect.FileDescriptor

var file_model_proto_rawDesc = []byte{
	0x0a, 0x0b, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x02, 0x70,
	0x62, 0x22, 0x7a, 0x0a, 0x05, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x12, 0x12, 0x0a, 0x04, 0x74, 0x79,
	0x70, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x04, 0x74, 0x79, 0x70, 0x65, 0x12, 0x1d,
	0x0a, 0x0a, 0x65, 0x6c, 0x65, 0x6d, 0x5f, 0x63, 0x6f, 0x75, 0x6e, 0x74, 0x18, 0x02, 0x20, 0x01,
	0x28, 0x03, 0x52, 0x09, 0x65, 0x6c, 0x65, 0x6d, 0x43, 0x6f, 0x75, 0x6e, 0x74, 0x12, 0x12, 0x0a,
	0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d,
	0x65, 0x12, 0x16, 0x0a, 0x06, 0x73, 0x68, 0x61, 0x70, 0x65, 0x73, 0x18, 0x04, 0x20, 0x03, 0x28,
	0x03, 0x52, 0x06, 0x73, 0x68, 0x61, 0x70, 0x65, 0x73, 0x12, 0x12, 0x0a, 0x04, 0x66, 0x69, 0x6c,
	0x65, 0x18, 0x05, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x66, 0x69, 0x6c, 0x65, 0x22, 0xb6, 0x01,
	0x0a, 0x05, 0x6c, 0x61, 0x79, 0x65, 0x72, 0x12, 0x14, 0x0a, 0x05, 0x63, 0x6c, 0x61, 0x73, 0x73,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x63, 0x6c, 0x61, 0x73, 0x73, 0x12, 0x12, 0x0a,
	0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d,
	0x65, 0x12, 0x21, 0x0a, 0x06, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x73, 0x18, 0x03, 0x20, 0x03, 0x28,
	0x0b, 0x32, 0x09, 0x2e, 0x70, 0x62, 0x2e, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x52, 0x06, 0x70, 0x61,
	0x72, 0x61, 0x6d, 0x73, 0x12, 0x27, 0x0a, 0x04, 0x61, 0x72, 0x67, 0x73, 0x18, 0x04, 0x20, 0x03,
	0x28, 0x0b, 0x32, 0x13, 0x2e, 0x70, 0x62, 0x2e, 0x6c, 0x61, 0x79, 0x65, 0x72, 0x2e, 0x41, 0x72,
	0x67, 0x73, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x52, 0x04, 0x61, 0x72, 0x67, 0x73, 0x1a, 0x37, 0x0a,
	0x09, 0x41, 0x72, 0x67, 0x73, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x12, 0x10, 0x0a, 0x03, 0x6b, 0x65,
	0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x6b, 0x65, 0x79, 0x12, 0x14, 0x0a, 0x05,
	0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x02, 0x52, 0x05, 0x76, 0x61, 0x6c,
	0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x22, 0x34, 0x0a, 0x0f, 0x6f, 0x70, 0x74, 0x69, 0x6d, 0x69,
	0x7a, 0x65, 0x72, 0x5f, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x12, 0x21, 0x0a, 0x06, 0x70, 0x61, 0x72,
	0x61, 0x6d, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x09, 0x2e, 0x70, 0x62, 0x2e, 0x70,
	0x61, 0x72, 0x61, 0x6d, 0x52, 0x06, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x73, 0x22, 0x68, 0x0a, 0x09,
	0x6f, 0x70, 0x74, 0x69, 0x6d, 0x69, 0x7a, 0x65, 0x72, 0x12, 0x14, 0x0a, 0x05, 0x63, 0x6c, 0x61,
	0x73, 0x73, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x63, 0x6c, 0x61, 0x73, 0x73, 0x12,
	0x18, 0x0a, 0x07, 0x6f, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0c,
	0x52, 0x07, 0x6f, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x12, 0x2b, 0x0a, 0x06, 0x70, 0x61, 0x72,
	0x61, 0x6d, 0x73, 0x18, 0x03, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x13, 0x2e, 0x70, 0x62, 0x2e, 0x6f,
	0x70, 0x74, 0x69, 0x6d, 0x69, 0x7a, 0x65, 0x72, 0x5f, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x52, 0x06,
	0x70, 0x61, 0x72, 0x61, 0x6d, 0x73, 0x22, 0x55, 0x0a, 0x03, 0x6e, 0x65, 0x74, 0x12, 0x21, 0x0a,
	0x06, 0x6c, 0x61, 0x79, 0x65, 0x72, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x09, 0x2e,
	0x70, 0x62, 0x2e, 0x6c, 0x61, 0x79, 0x65, 0x72, 0x52, 0x06, 0x6c, 0x61, 0x79, 0x65, 0x72, 0x73,
	0x12, 0x2b, 0x0a, 0x09, 0x6f, 0x70, 0x74, 0x69, 0x6d, 0x69, 0x7a, 0x65, 0x72, 0x18, 0x02, 0x20,
	0x01, 0x28, 0x0b, 0x32, 0x0d, 0x2e, 0x70, 0x62, 0x2e, 0x6f, 0x70, 0x74, 0x69, 0x6d, 0x69, 0x7a,
	0x65, 0x72, 0x52, 0x09, 0x6f, 0x70, 0x74, 0x69, 0x6d, 0x69, 0x7a, 0x65, 0x72, 0x42, 0x06, 0x5a,
	0x04, 0x2e, 0x3b, 0x70, 0x62, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_model_proto_rawDescOnce sync.Once
	file_model_proto_rawDescData = file_model_proto_rawDesc
)

func file_model_proto_rawDescGZIP() []byte {
	file_model_proto_rawDescOnce.Do(func() {
		file_model_proto_rawDescData = protoimpl.X.CompressGZIP(file_model_proto_rawDescData)
	})
	return file_model_proto_rawDescData
}

var file_model_proto_msgTypes = make([]protoimpl.MessageInfo, 6)
var file_model_proto_goTypes = []interface{}{
	(*Param)(nil),          // 0: pb.param
	(*Layer)(nil),          // 1: pb.layer
	(*OptimizerParam)(nil), // 2: pb.optimizer_param
	(*Optimizer)(nil),      // 3: pb.optimizer
	(*Net)(nil),            // 4: pb.net
	nil,                    // 5: pb.layer.ArgsEntry
}
var file_model_proto_depIdxs = []int32{
	0, // 0: pb.layer.params:type_name -> pb.param
	5, // 1: pb.layer.args:type_name -> pb.layer.ArgsEntry
	0, // 2: pb.optimizer_param.params:type_name -> pb.param
	2, // 3: pb.optimizer.params:type_name -> pb.optimizer_param
	1, // 4: pb.net.layers:type_name -> pb.layer
	3, // 5: pb.net.optimizer:type_name -> pb.optimizer
	6, // [6:6] is the sub-list for method output_type
	6, // [6:6] is the sub-list for method input_type
	6, // [6:6] is the sub-list for extension type_name
	6, // [6:6] is the sub-list for extension extendee
	0, // [0:6] is the sub-list for field type_name
}

func init() { file_model_proto_init() }
func file_model_proto_init() {
	if File_model_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_model_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Param); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_model_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Layer); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_model_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*OptimizerParam); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_model_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Optimizer); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_model_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Net); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_model_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   6,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_model_proto_goTypes,
		DependencyIndexes: file_model_proto_depIdxs,
		MessageInfos:      file_model_proto_msgTypes,
	}.Build()
	File_model_proto = out.File
	file_model_proto_rawDesc = nil
	file_model_proto_goTypes = nil
	file_model_proto_depIdxs = nil
}
