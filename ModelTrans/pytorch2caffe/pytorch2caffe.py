import torch
import torch.nn as nn
import traceback
from . import caffe_net, layer_param
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import numpy as np
from torch.nn.modules.utils import _list_with_default
"""
How to support a new layer type:
 layer_name=log.add_layer(layer_type_name)
 top_blobs=log.add_blobs(<output of that layer>)
 layer=caffe_net.Layer_param(xxx)
 <set layer parameters>
 [<layer.add_data(*datas)>]
 log.cnet.add_layer(layer)
 
Please MUTE the inplace operations to avoid not find in graph

注意：只有torch.nn.functional中的函数才能转换为caffe中的层
"""

# TODO: support the inplace output of the layers


class Blob_LOG():
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


NET_INITTED = False


# 转换原理解析：通过记录
class TransLog(object):
    def __init__(self):
        """
        doing init() with inputs Variable before using it
        """
        self.layers = {}
        self.detail_layers = {}
        self.detail_blobs = {}
        self._blobs = Blob_LOG()
        self._blobs_data = []
        self.cnet = caffe_net.Caffemodel('')
        self.debug = True

    def init(self, inputs):
        """
        :param inputs: is a list of input variables
        """
        self.add_blobs(inputs)

    def clear(self):
        self.layers = {}
        self.detail_layers = {}
        self.detail_blobs = {}
        self._blobs = Blob_LOG()
        self._blobs_data = []
        self.cnet = caffe_net.Caffemodel('')
        self.debug = True

    def add_layer(self, name='layer'):
        if name in self.layers:
            return self.layers[name]
        if name not in self.detail_layers.keys():
            self.detail_layers[name] = 0
        self.detail_layers[name] += 1
        name = '{}{}'.format(name, self.detail_layers[name])
        self.layers[name] = name
        if self.debug:
            print("{} was added to layers".format(self.layers[name]))
        return self.layers[name]

    def add_blobs(self, blobs, name='blob', with_num=True):
        rst = []
        for blob in blobs:
            self._blobs_data.append(
                blob)  # to block the memory address be rewrited
            blob_id = int(id(blob))
            if name not in self.detail_blobs.keys():
                self.detail_blobs[name] = 0
            self.detail_blobs[name] += 1
            if with_num:
                rst.append('{}{}'.format(name, self.detail_blobs[name]))
            else:
                rst.append('{}'.format(name))
            if self.debug:
                print("{}:{} was added to blobs".format(blob_id, rst[-1]))
            # print('Add blob {} : {}'.format(rst[-1].center(21),blob.size()))
            self._blobs[blob_id] = rst[-1]
        return rst

    def blobs(self, var):
        var = id(var)
        # if self.debug:
        #     print("{}:{} getting".format(var, self._blobs[var]))
        try:
            return self._blobs[var]
        except:
            print("WARNING: CANNOT FOUND blob {}".format(var))
            return None


log = TransLog()

layer_names = {}


def _conv2d(raw,
            input,
            weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1):
    print("------------------_conv2d----------------------")
    print('conv: ', log.blobs(input))
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    name = log.add_layer(name='conv')
    log.add_blobs([x], name='conv_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='Convolution',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    layer.conv_param(x.size()[1],
                     weight.size()[2:],
                     stride=_pair(stride),
                     pad=_pair(padding),
                     dilation=_pair(dilation),
                     bias_term=bias is not None,
                     groups=groups)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _conv_transpose2d(raw,
                      input,
                      weight,
                      bias=None,
                      stride=1,
                      padding=0,
                      output_padding=0,
                      groups=1,
                      dilation=1):
    print("------------------_conv_transpose2d----------------------")
    x = raw(input, weight, bias, stride, padding, output_padding, groups,
            dilation)
    name = log.add_layer(name='conv_transpose')
    log.add_blobs([x], name='conv_transpose_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='Deconvolution',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    layer.conv_param(x.size()[1],
                     weight.size()[2:],
                     stride=_pair(stride),
                     pad=_pair(padding),
                     dilation=_pair(dilation),
                     bias_term=bias is not None,
                     groups=groups)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _linear(raw, input, weight, bias=None):
    print("------------------_linear----------------------")
    x = raw(input, weight, bias)
    layer_name = log.add_layer(name='fc')
    top_blobs = log.add_blobs([x], name='fc_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='InnerProduct',
                                  bottom=[log.blobs(input)],
                                  top=top_blobs)
    layer.fc_param(x.size()[1], has_bias=bias is not None)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _split(raw, tensor, split_size, dim=0):
    # split in pytorch is slice in caffe
    x = raw(tensor, split_size, dim)
    layer_name = log.add_layer('split')
    top_blobs = log.add_blobs(x, name='split_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Slice',
                                  bottom=[log.blobs(tensor)],
                                  top=top_blobs)
    slice_num = int(np.floor(tensor.size()[dim] / split_size))
    slice_param = caffe_net.pb.SliceParameter(
        axis=dim, slice_point=[split_size * i for i in range(1, slice_num)])
    layer.param.slice_param.CopyFrom(slice_param)
    log.cnet.add_layer(layer)
    return x


def _pool(type, raw, input, x, kernel_size, stride, padding, ceil_mode):
    # TODO dilation,ceil_mode,return indices
    layer_name = log.add_layer(name='{}_pool'.format(type))
    top_blobs = log.add_blobs([x], name='{}_pool_blob'.format(type))
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Pooling',
                                  bottom=[log.blobs(input)],
                                  top=top_blobs)
    # TODO w,h different kernel, stride and padding
    # processing ceil mode
    print('ceil', ceil_mode)
    # layer.pool_param(kernel_size=kernel_size,
    #                  stride=kernel_size if stride is None else stride,
    #                  pad=padding,
    #                  type=type.upper(),
    #                  ceil_mode=ceil_mode)
    # log.cnet.add_layer(layer)
    if ceil_mode == False and stride is not None:
        oheight = (input.size()[2] - _pair(kernel_size)[0] +
                   2 * _pair(padding)[0]) % (_pair(stride)[0])
        owidth = (input.size()[3] - _pair(kernel_size)[1] +
                  2 * _pair(padding)[1]) % (_pair(stride)[1])
        if oheight != 0 or owidth != 0:
                caffe_out = raw(input,
                            kernel_size,
                            stride,
                            padding,
                            ceil_mode=False)
                print(
                    "WARNING: the output shape miss match at {}: "
                    "input {} output---Pytorch:{}---Caffe:{}\n"
                    "This is caused by the different implementation that ceil mode in caffe and the floor mode in pytorch.\n"
                    "You can add the clip layer in caffe prototxt manually if shape mismatch error is caused in caffe. "
                    .format(layer_name, input.size(), x.size(), caffe_out.size()))
                if(abs(np.array(x.size())[3]-np.array(caffe_out.size())[3])==1):
                    layer.pool_param(kernel_size=kernel_size,
                        stride=kernel_size if stride is None else stride,
                        pad=padding,
                        type=type.upper(),
                        ceil_mode=True)
                    log.cnet.add_layer(layer)
                    print("已修正---Pytorch:{}---Caffe:{}\n".format(x.size(), caffe_out.size()))
                else:
                    layer.pool_param(kernel_size=kernel_size,
                    stride=kernel_size if stride is None else stride,
                    pad=padding,
                    type=type.upper(),
                    ceil_mode=False)
                    print('kernel_size', kernel_size)
                    print('padding_size', padding)
                    print('stride_size', stride)
                    log.cnet.add_layer(layer)
        else:
                layer.pool_param(kernel_size=kernel_size,
                    stride=kernel_size if stride is None else stride,
                    pad=padding,
                    type=type.upper(),
                    ceil_mode=False)
                log.cnet.add_layer(layer)
def _max_pool2d(raw,
                input,
                kernel_size,
                stride=None,
                padding=0,
                dilation=1,
                ceil_mode=False,
                return_indices=False):
    print("------------------_max_pool2d----------------------")
    x = raw(input, kernel_size, stride, padding, dilation, ceil_mode,
            return_indices)
    _pool('max', raw, input, x, kernel_size, stride, padding, ceil_mode)
    return x


def _avg_pool2d(raw,
                input,
                kernel_size,
                stride=None,
                padding=0,
                ceil_mode=False,
                count_include_pad=True):
    print("------------------_avg_pool2d----------------------")
    x = raw(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    _pool('ave', raw, input, x, kernel_size, stride, padding, ceil_mode)
    return x


def _adaptive_avg_pool2d(raw, input, output_size):
    print("------------------_adaptive_avg_pool2d----------------------")
    _output_size = _list_with_default(output_size, input.size())
    # print('_output_size',_output_size)
    x = raw(input, _output_size)
    # print('in_size',input.size())
    # print('out_size',output_size)
    if isinstance(output_size, int):
        outputsz=np.array([output_size,output_size])
    else:
        print(type(_output_size))
        outputsz=np.array(_output_size)
    inputsz = np.array(input.shape[2:])
    # print(inputsz)
    # print(outputsz)
    stridesz = inputsz//outputsz
    kernelsz = inputsz-(outputsz-1)*stridesz
    if (kernelsz[0]==kernelsz[1]).all():
        kernelsz=int(kernelsz[0])
    else:
        kernelsz=list(kernelsz)
    if (stridesz[0]==stridesz[1]).all():
        stridesz=int(stridesz[0])
    else:
        stridesz=list(stridesz)
    print(kernelsz,stridesz)
    _pool('ave', raw, input,x, kernelsz,stridesz, 0, False)
    # print('adaptive_avg_pool2d转换后shape=',x.size())
   
    return x

def _squeeze(input, *args):
    x=raw_squeeze(input, *args)
    print("------------------_squeeze----------------------")
    if not NET_INITTED:
        return x
    dims = list(input.shape)
    if len(args) == 0:
        start = 0
    else:
        start = args[0]

    layer_name = log.add_layer(name='squeeze')
    top_blobs = log.add_blobs([x], name='view_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Reshape',
                                  bottom=[log.blobs(input)],
                                  top=top_blobs)
    # TODO: reshpae added to nn_tools layer
    #print(args[0])

    #dims = [dims[i] for i in range(-1)]
    if dims[start]==1:
        dims.pop(start)
    # print("变换后dim",dims)
    dims[0] = 0# the first dim should be batch_size
    layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))
    log.cnet.add_layer(layer)
    return x

def _max(raw, *args):
    print("------------------_max----------------------")
    x = raw(*args)
    if len(args) == 1:
        # TODO max in one tensor
        assert NotImplementedError
    else:
        bottom_blobs = []
        for arg in args:
            bottom_blobs.append(log.blobs(arg))
        layer_name = log.add_layer(name='max')
        top_blobs = log.add_blobs([x], name='max_blob')
        layer = caffe_net.Layer_param(name=layer_name,
                                      type='Eltwise',
                                      bottom=bottom_blobs,
                                      top=top_blobs)
        layer.param.eltwise_param.operation = 2
        log.cnet.add_layer(layer)
    return x


def _cat(raw, inputs, dim=0):
    x = raw(inputs, dim)
    bottom_blobs = []
    for input in inputs:
        bottom_blobs.append(log.blobs(input))
    layer_name = log.add_layer(name='cat')
    top_blobs = log.add_blobs([x], name='cat_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Concat',
                                  bottom=bottom_blobs,
                                  top=top_blobs)
    layer.param.concat_param.axis = dim
    log.cnet.add_layer(layer)
    return x


def _dropout(raw, input, p=0.5, training=False, inplace=False):
    x = raw(input, p, training, inplace)
    bottom_blobs = [log.blobs(input)]
    layer_name = log.add_layer(name='dropout')
    top_blobs = log.add_blobs([x], name=bottom_blobs[0], with_num=False)
    print(bottom_blobs, top_blobs)
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Dropout',
                                  bottom=bottom_blobs,
                                  top=top_blobs)
    layer.param.dropout_param.dropout_ratio = p
    layer.param.include.extend([caffe_net.pb.NetStateRule(phase=0)
                                ])  # 1 for test, 0 for train
    log.cnet.add_layer(layer)
    return x


def _flatten(raw, input, start_dim=0, end_dim=-1):
    print("------------------_flatten----------------------")
    x = raw(input, start_dim, end_dim)
    bottom_blobs = [log.blobs(input)]
    layer_name = log.add_layer(name='flatten')
    top_blobs = log.add_blobs([x], name='flatten_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Flatten',
                                  bottom=bottom_blobs,
                                  top=top_blobs)
    layer.param.flatten_param.axis = start_dim
    layer.param.flatten_param.end_axis = end_dim
    log.cnet.add_layer(layer)
    return x


def _bmm(raw, input1, input2):
    print("------------------_bmm----------------------")
    x = raw(input1, input2)
    bottom_blobs = [log.blobs(input1), log.blobs(input2)]
    layer_name = log.add_layer(name='bmm')
    top_blobs = log.add_blobs([x], name='bmm_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='BatchMatMul',
                                  bottom=bottom_blobs,
                                  top=top_blobs)
    log.cnet.add_layer(layer)
    return x

def _threshold(raw, input, threshold, value, inplace=False):
    # for threshold or relu
    print("------------------_threshold----------------------")
    if threshold == 0 and value == 0:
        x = raw(input, threshold, value, inplace)
        bottom_blobs = [log.blobs(input)]
        name = log.add_layer(name='relu')
        log.add_blobs([x], name='relu_blob')
        layer = caffe_net.Layer_param(name=name,
                                      type='ReLU',
                                      bottom=bottom_blobs,
                                      top=[log.blobs(x)])
        log.cnet.add_layer(layer)
        return x
    if value != 0:
        raise NotImplemented("value !=0 not implemented in caffe")
    x = raw(input, input, threshold, value, inplace)
    bottom_blobs = [log.blobs(input)]
    layer_name = log.add_layer(name='threshold')
    top_blobs = log.add_blobs([x], name='threshold_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Threshold',
                                  bottom=bottom_blobs,
                                  top=top_blobs)
    layer.param.threshold_param.threshold = threshold
    log.cnet.add_layer(layer)
    return x


def _relu(raw, input, inplace=False):
    # for threshold or prelu
    print("------------------_relu----------------------")
    x = raw(input, False)
    name = log.add_layer(name='relu')
    log.add_blobs([x], name='relu_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='ReLU',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


def _prelu(raw, input, weight):
    # for threshold or prelu
    x = raw(input, weight)
    bottom_blobs = [log.blobs(input)]
    name = log.add_layer(name='prelu')
    log.add_blobs([x], name='prelu_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='PReLU',
                                  bottom=bottom_blobs,
                                  top=[log.blobs(x)])
    if weight.size()[0] == 1:
        layer.param.prelu_param.channel_shared = True
        layer.add_data(weight.cpu().data.numpy()[0])
    else:
        layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    x = raw(input, negative_slope)
    name = log.add_layer(name='leaky_relu')
    log.add_blobs([x], name='leaky_relu_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='ReLU',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    layer.param.relu_param.negative_slope = negative_slope
    log.cnet.add_layer(layer)
    return x

# 新增silu对应接口
def _silu(raw, input, inplace=False):
    x = input * torch.sigmoid(input)
    # x = raw(input, negative_slope)
    # name = log.add_layer(name='leaky_relu')
    # log.add_blobs([x], name='leaky_relu_blob')
    # layer = caffe_net.Layer_param(name=name,
    #                               type='ReLU',
    #                               bottom=[log.blobs(input)],
    #                               top=[log.blobs(x)])
    # layer.param.relu_param.negative_slope = negative_slope
    # log.cnet.add_layer(layer)
    return x


def _tanh(raw, input):
    # for tanh activation
    x = raw(input)
    name = log.add_layer(name='tanh')
    log.add_blobs([x], name='tanh_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='TanH',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


def _softmax(raw, input, dim=None, _stacklevel=3):
    # for F.softmax
    print("------------------_softmax----------------------")
    x = raw(input, dim=dim)
    if dim is None:
        dim = F._get_softmax_dim('softmax', input.dim(), _stacklevel)
    bottom_blobs = [log.blobs(input)]
    name = log.add_layer(name='softmax')
    log.add_blobs([x], name='softmax_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='Softmax',
                                  bottom=bottom_blobs,
                                  top=[log.blobs(x)])
    layer.param.softmax_param.axis = dim
    log.cnet.add_layer(layer)
    return x


def _batch_norm(raw,
                input,
                running_mean,
                running_var,
                weight=None,
                bias=None,
                training=False,
                momentum=0.1,
                eps=1e-5):
    # because the runing_mean and runing_var will be changed after the _batch_norm operation, we first save the parameters

    x = raw(input, running_mean, running_var, weight, bias, training, momentum,eps)
    bottom_blobs = [log.blobs(input)]
    layer_name1 = log.add_layer(name='batch_norm')
    top_blobs = log.add_blobs([x], name='batch_norm_blob')
    layer1 = caffe_net.Layer_param(name=layer_name1,
                                   type='BatchNorm',
                                   bottom=bottom_blobs,
                                   top=top_blobs)
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0, eps=eps)
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
        running_mean_clone = running_mean.clone()
        running_var_clone = running_var.clone()
        # 这里加入了权重信息
        layer1.add_data(running_mean_clone.cpu().numpy(),
                        running_var_clone.cpu().numpy(), np.array([1.0]))
    log.cnet.add_layer(layer1)
    if weight is not None and bias is not None:
        layer_name2 = log.add_layer(name='bn_scale')
        layer2 = caffe_net.Layer_param(name=layer_name2,
                                       type='Scale',
                                       bottom=top_blobs,
                                       top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        log.cnet.add_layer(layer2)
    return x


# def _instance_norm(raw,
#                    input,
#                    running_mean=None,
#                    running_var=None,
#                    weight=None,
#                    bias=None,
#                    use_input_stats=True,
#                    momentum=0.1,
#                    eps=1e-5):
#     # TODO: the batch size!=1 view operations
#     print(
#         "WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1"
#     )
#     if running_var is not None or weight is not None:
#         # TODO: the affine=True or track_running_stats=True case
#         raise NotImplementedError(
#             "not implement the affine=True or track_running_stats=True case InstanceNorm"
#         )
#     x = torch.batch_norm(input, weight, bias, running_mean, running_var,
#                          use_input_stats, momentum, eps,
#                          torch.backends.cudnn.enabled)
#     bottom_blobs = [log.blobs(input)]
#     layer_name1 = log.add_layer(name='instance_norm')
#     top_blobs = log.add_blobs([x], name='instance_norm_blob')
#     layer1 = caffe_net.Layer_param(name=layer_name1,
#                                    type='BatchNorm',
#                                    bottom=bottom_blobs,
#                                    top=top_blobs)
#     if running_mean is None or running_var is None:
#         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         # not use global_stats, normalization is performed over the current mini-batch
#         layer1.batch_norm_param(use_global_stats=0, eps=eps)
#         running_mean = torch.zeros(input.size()[1])
#         running_var = torch.ones(input.size()[1])
#     else:
#         layer1.batch_norm_param(use_global_stats=1, eps=eps)
#     running_mean_clone = running_mean.clone()
#     running_var_clone = running_var.clone()
#     layer1.add_data(running_mean_clone.cpu().numpy(),
#                     running_var_clone.cpu().numpy(), np.array([1.0]))
#     log.cnet.add_layer(layer1)
#     if weight is not None and bias is not None:
#         layer_name2 = log.add_layer(name='bn_scale')
#         layer2 = caffe_net.Layer_param(name=layer_name2,
#                                        type='Scale',
#                                        bottom=top_blobs,
#                                        top=top_blobs)
#         layer2.param.scale_param.bias_term = True
#         layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
#         log.cnet.add_layer(layer2)
#     return x


#upsample layer

# 重写instance_norm函数，使得这个函数对应那个层

# 修改这个instance_norm，使得in变成relu和in_scale两层叠加
# 这个是基于relu手动修改这个type
# def _instance_norm(raw,
#                    input,
#                    running_mean=None,
#                    running_var=None,
#                    weight=None,
#                    bias=None,
#                    use_input_stats=True,
#                    momentum=0.1,
#                    eps=1e-5):
#     # TODO: the batch size!=1 view operations
#     print(
#         "WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1"
#     )
#     if running_var is not None or running_mean is not None:
#         # TODO: the affine=True or track_running_stats=True case
#         raise NotImplementedError(
#             "not implement the affine=True or track_running_stats=True case InstanceNorm"
#         )
#     print('in层的eps为:{0}\n'.format(eps))
#     x = torch.instance_norm(input, weight, bias, running_mean, running_var,
#                          use_input_stats, momentum, eps,
#                          torch.backends.cudnn.enabled)
#     bottom_blobs = [log.blobs(input)]
#     layer_name1 = log.add_layer(name='instance_norm')
#     top_blobs = log.add_blobs([x], name='instance_norm_blob')
#     layer1 = caffe_net.Layer_param(name=layer_name1,
#                                    type='ReLU',
#                                    bottom=bottom_blobs,
#                                    top=top_blobs)
#     log.cnet.add_layer(layer1)
#     if weight is not None and bias is not None:
#         layer_name2 = log.add_layer(name='in_scale')
#         layer2 = caffe_net.Layer_param(name=layer_name2,
#                                        type='Scale',
#                                        bottom=top_blobs,
#                                        top=top_blobs)
#         layer2.param.scale_param.bias_term = True
#         layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
#         log.cnet.add_layer(layer2)
#     return x

# 尝试时间:2021.5.12，本次使用protobuf解析来进行脚本添加
# 再次修改：2021.5.13，本次打算在scale也添加进算子内部
# 再次修改：2021.5.18，本次通过控制输入是否是NCHW还是NC1HWC0，来控制算子速度
# def _instance_norm(raw,
#                    input,
#                    running_mean=None,
#                    running_var=None,
#                    weight=None,
#                    bias=None,
#                    use_input_stats=True,
#                    momentum=0.1,
#                    eps=1e-5):
#     # TODO: the batch size!=1 view operations
#     print(
#         "WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1"
#     )
#     if running_var is not None or running_mean is not None:
#         # TODO: the affine=True or track_running_stats=True case
#         raise NotImplementedError(
#             "not implement the affine=True or track_running_stats=True case InstanceNorm"
#         )
#     # print('in层的eps为:{0}\n'.format(eps))
#     x = torch.instance_norm(input, weight, bias, running_mean, running_var,
#                          use_input_stats, momentum, eps,
#                          torch.backends.cudnn.enabled)
#     bottom_blobs = [log.blobs(input)]
#     layer_name1 = log.add_layer(name='instance_norm')
#     top_blobs = log.add_blobs([x], name='instance_norm_blob')
#     n,c,h,w = input.shape
#     if (h*w) > (24*24):
#         typename = 'Inpzk'
#     else:
#         typename = 'Inpzkv2'
#     layer1 = caffe_net.Layer_param(name=layer_name1,
#                                        type=typename,
#                                        bottom=bottom_blobs,
#                                        top=top_blobs)
#     layer1.param.inpzk_param.eps = eps
#
#     # 加入偏置信息，看是否出错
#     if weight is not None and bias is not None:
#         layer1.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
#     else:
#         bias = torch.zeros(input.size()[1])
#         weight = torch.ones(input.size()[1])
#         weight_clone = weight.clone()
#         bias_clone = bias.clone()
#         layer1.add_data(weight_clone.cpu().data.numpy(), bias_clone.cpu().data.numpy())
#     log.cnet.add_layer(layer1)
#     return x

# 再次修改：2021.6.21，本次增加了TIK开发的in算子层，尝试intik
# 偏置加入到后面
def _instance_norm(raw,
                   input,
                   running_mean=None,
                   running_var=None,
                   weight=None,
                   bias=None,
                   use_input_stats=True,
                   momentum=0.1,
                   eps=1e-5):
    # TODO: the batch size!=1 view operations
    print(
        "WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1"
    )
    if running_var is not None or running_mean is not None:
        # TODO: the affine=True or track_running_stats=True case
        raise NotImplementedError(
            "not implement the affine=True or track_running_stats=True case InstanceNorm"
        )
    # print('in层的eps为:{0}\n'.format(eps))
    x = torch.instance_norm(input, weight, bias, running_mean, running_var,
                         use_input_stats, momentum, eps,
                         torch.backends.cudnn.enabled)
    bottom_blobs = [log.blobs(input)]
    layer_name1 = log.add_layer(name='instance_norm')
    top_blobs = log.add_blobs([x], name='instance_norm_blob')
    typename = 'Intik'
    layer1 = caffe_net.Layer_param(name=layer_name1,
                                       type=typename,
                                       bottom=bottom_blobs,
                                       top=top_blobs)
    layer1.param.intik_param.eps = eps
    log.cnet.add_layer(layer1)
    # 加入偏置信息，看是否出错
    if weight is not None and bias is not None:
        layer_name2 = log.add_layer(name='in_scale')
        layer2 = caffe_net.Layer_param(name=layer_name2,
                                       type='Scale',
                                       bottom=top_blobs,
                                       top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        log.cnet.add_layer(layer2)
    return x

def _ReflectionPad2d(raw,
                    input,
                    pad,
                    mode ="constant",
                    value = 0
                     ):
    if not(mode == "reflect"):
        return raw(input, pad, mode, value)
    # print('in层的eps为:{0}\n'.format(eps))
    x = raw(input, pad, mode, value)
    bottom_blobs = [log.blobs(input)]
    layer_name1 = log.add_layer(name='reflectionpad2d')
    top_blobs = log.add_blobs([x], name='reflectionpad2d_blob')
    typename = 'Reduction'
    layer = caffe_net.Layer_param(name=layer_name1,
                                       type=typename,
                                       bottom=bottom_blobs,
                                       top=top_blobs)
    layer.param.reduction_param.operation = 4
    if isinstance(pad, list) or isinstance(pad, tuple):
        pad = pad[0]
    layer.param.reduction_param.axis = pad
    log.cnet.add_layer(layer)
    return x

# 这里为了保证vit能够识别出layernorm层，进行了暂时的替换
def _layer_norm(raw, input,
    normalized_shape,
    weight = None,
    bias = None,
    eps = 1e-5):
    print(
        "WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1"
    )
    if weight is  None and bias is None:
        # TODO: the affine=True or track_running_stats=True case
        raise NotImplementedError(
            "not implement both weight and bias is None case LayerNorm"
        )
    # print('in层的eps为:{0}\n'.format(eps))
    x = torch.layer_norm(input, normalized_shape, weight, bias, eps,
                         torch.backends.cudnn.enabled)
    bottom_blobs = [log.blobs(input)]
    layer_name1 = log.add_layer(name='LayerNorm')
    top_blobs = log.add_blobs([x], name='instance_norm_blob')
    typename = 'Intik'
    layer1 = caffe_net.Layer_param(name=layer_name1,
                                       type=typename,
                                       bottom=bottom_blobs,
                                       top=top_blobs)
    layer1.param.intik_param.eps = eps
    log.cnet.add_layer(layer1)
    # 加入偏置信息，看是否出错
    if weight is not None and bias is not None:
        layer_name2 = log.add_layer(name='LayerNorm-Scale')
        layer2 = caffe_net.Layer_param(name=layer_name2,
                                       type='Scale',
                                       bottom=top_blobs,
                                       top=top_blobs)
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        log.cnet.add_layer(layer2)
    return x

def _interpolate(raw,
                 input,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
    # 定义的参数包括 scale,即输出与输入的尺寸比例,如 2;scale_h、scale_w,
    # 同 scale,分别为 h、w 方向上的尺寸比例;pad_out_h、pad_out_w,仅在 scale 为 2 时
    # 有用,对输出进行额外 padding 在 h、w 方向上的数值;upsample_h、upsample_w,输
    # 出图像尺寸的数值。在 Upsample 的相关代码中,推荐仅仅使用 upsample_h、
    # upsample_w 准确定义 Upsample 层的输出尺寸,其他所有的参数都不推荐继续使用。
    # for nearest _interpolate

    x = raw(input, size, scale_factor, mode)
    if mode == "nearest" and align_corners == None:
        layer_name = log.add_layer(name='upsample')
        top_blobs = log.add_blobs([x], name='upsample_blob'.format(type))
        layer = caffe_net.Layer_param(name=layer_name,
                                      type='Upsample',
                                      bottom=[log.blobs(input)],
                                      top=top_blobs)

        layer.upsample_param(size=(input.size(2), input.size(3)),
                             scale_factor=scale_factor)
        log.cnet.add_layer(layer)
    else:
        layer_name = log.add_layer(name='interp')
        top_blobs = log.add_blobs([x], name='interp_blob'.format(type))
        layer = caffe_net.Layer_param(name=layer_name,
                                      type='Interp',
                                      bottom=[log.blobs(input)],
                                      top=top_blobs)

        # layer.upsample_param(size =(input.size(2),input.size(3)), scale_factor= scale_factor)
        layer.param.interp_param.height = x.size(2)
        layer.param.interp_param.width = x.size(3)
        log.cnet.add_layer(layer)
    # else:
    # raise NotImplementedError("not implement F.interpolate totoaly")
    return x


#sigmid layer
def _sigmoid(raw, input):
    # Applies the element-wise function:
    #
    # Sigmoid(x)= 1/(1+exp(−x)）
    #
    # ​
    print("------------------_sigmoid----------------------")
    x = raw(input)
    name = log.add_layer(name='sigmoid')
    log.add_blobs([x], name='sigmoid_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='Sigmoid',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


#tanh layer
def _tanh(raw, input):
    # Applies the element-wise function:
    #
    # torch.nn.Tanh
    #
    # ​
    x = raw(input)
    name = log.add_layer(name='tanh')
    log.add_blobs([x], name='tanh_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='TanH',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


def _hardtanh(raw, input, min_val, max_val, inplace):
    # Applies the element-wise function:
    #
    # torch.nn.ReLu6
    #
    # ​
    print('relu6: ', log.blobs(input))
    x = raw(input, min_val, max_val)
    name = log.add_layer(name='relu6')
    log.add_blobs([x], name='relu6_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='ReLU6',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


#L2Norm layer
def _l2Norm(raw, input, weight, eps):
    # Applies the element-wise function:
    #
    # L2Norm in vgg_ssd
    #
    # ​
    x = raw(input, weight, eps)
    name = log.add_layer(name='normalize')
    log.add_blobs([x], name='normalize_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='Normalize',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    layer.norm_param(eps)

    layer.add_data(weight.cpu().data.numpy())
    log.cnet.add_layer(layer)
    return x


def _div(raw, inputs, inputs2):
    x = raw(inputs, inputs2)
    log.add_blobs([x], name='div_blob')
    return x


# ----- for Variable operations --------


def _view(input, *args):
    x = raw_view(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='view')
    top_blobs = log.add_blobs([x], name='view_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Reshape',
                                  bottom=[log.blobs(input)],
                                  top=top_blobs)
    # TODO: reshpae added to nn_tools layer
    dims = list(args)
    dims[0] = 0  # the first dim should be batch_size
    layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))
    log.cnet.add_layer(layer)
    return x


def _mean(input, *args, **kwargs):
    x = raw_mean(input, *args, **kwargs)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='mean')
    top_blobs = log.add_blobs([x], name='mean_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Reduction',
                                  bottom=[log.blobs(input)],
                                  top=top_blobs)
    if len(args) == 1:
        dim = args[0]
    elif 'dim' in kwargs:
        dim = kwargs['dim']
    else:
        raise NotImplementedError('mean operation must specify a dim')
    layer.param.reduction_param.operation = 4
    layer.param.reduction_param.axis = dim
    log.cnet.add_layer(layer)
    return x


def _add(input, *args):
    x = raw__add__(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    if log.blobs(args[0]) == None:
        log.add_blobs([args[0]], name='extra_blob')
    else:
        layer = caffe_net.Layer_param(
            name=layer_name,
            type='Eltwise',
            bottom=[log.blobs(input), log.blobs(args[0])],
            top=top_blobs)
        layer.param.eltwise_param.operation = 1  # sum is 1
        log.cnet.add_layer(layer)
    return x


def _iadd(input, *args):
    x = raw__iadd__(input, *args)
    if not NET_INITTED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    layer = caffe_net.Layer_param(
        name=layer_name,
        type='Eltwise',
        bottom=[log.blobs(input), log.blobs(args[0])],
        top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x


def _sub(input, *args):
    x = raw__sub__(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='sub')
    top_blobs = log.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(
        name=layer_name,
        type='Eltwise',
        bottom=[log.blobs(input), log.blobs(args[0])],
        top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    log.cnet.add_layer(layer)
    return x


def _isub(input, *args):
    x = raw__isub__(input, *args)
    if not NET_INITTED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='sub')
    top_blobs = log.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(
        name=layer_name,
        type='Eltwise',
        bottom=[log.blobs(input), log.blobs(args[0])],
        top=top_blobs)
    layer.param.eltwise_param.operation = 1  # sum is 1
    log.cnet.add_layer(layer)
    return x

def get_shape_size(shape:list):
    size = 1
    if len(shape) == 0:
        size = 0
    else:
        for i in shape:
            size = size * i
    return size

def broadcast_shape(input1_shape:list, input2_shape:list):
    # max_shape = []
    if get_shape_size(input1_shape) > get_shape_size(input2_shape):
        return input1_shape
    else:
        return input2_shape


def _mul(input, *args):
    x = raw__mul__(input, *args)
    if not NET_INITTED:
        return x
    # 增加广播操作
    # from functools import reduce
    input1_shape = list(input.shape)
    input2_shape = list(args[0].shape)
    input1_size = get_shape_size(input1_shape)
    input2_size = get_shape_size(input2_shape)
    max_shape = broadcast_shape(input1_shape, input2_shape)
    eltwise_bottom = []
    broadcast_tensor = None
    if not (input1_size == input2_size):
        # 不等于的时候，才需要进行广播
        if input1_size > input2_size:
            # 广播args[0]
            broadcast_tensor = args[0].clone()
            # eltwise_bottom = [log.blobs(input), log.blobs(broadcast_tensor)]
        else:
            broadcast_tensor = input.clone()
            # eltwise_bottom = [log.blobs(broadcast_tensor), log.blobs(args[0])]

        # 广播的算子
        broadcast_layer_name = log.add_layer(name='broadcast_dsl')
        top_blobs = log.add_blobs([broadcast_tensor], name='broadcast_dsl_blob')
        broadcast_layer = caffe_net.Layer_param(
            name=broadcast_layer_name,
            type='BroadcastDsl',
            bottom=[log.blobs(input), log.blobs(args[0])],
            top=top_blobs)
        log.cnet.add_layer(broadcast_layer)
        if input1_size > input2_size:
            # 广播args[0]
            eltwise_bottom = [log.blobs(input), log.blobs(broadcast_tensor)]
        else:
            eltwise_bottom = [log.blobs(broadcast_tensor), log.blobs(args[0])]
    else:
        eltwise_bottom = [log.blobs(input), log.blobs(args[0])]
    # 原始的算子
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(
        name=layer_name,
        type='Eltwise',
        bottom=eltwise_bottom,
        top=top_blobs)
    layer.param.eltwise_param.operation = 0  # product is 1
    log.cnet.add_layer(layer)
    return x


def _imul(input, *args):
    x = raw__imul__(input, *args)
    if not NET_INITTED:
        return x
    x = x.clone()
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(
        name=layer_name,
        type='Eltwise',
        bottom=[log.blobs(input), log.blobs(args[0])],
        top=top_blobs)
    layer.param.eltwise_param.operation = 0  # product is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    log.cnet.add_layer(layer)
    return x


#Permute layer
def _permute(input, *args):
    x = raw__permute__(input, *args)
    name = log.add_layer(name='permute')
    log.add_blobs([x], name='permute_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='Permute',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    # 这里是因为默认是继续了图片处理，所以这里认为permute都是四维数据
    # 暂且为了vit能够正常转换成功，这里增加了判断机制来适应三维数据
    order1 = args[0]
    order2 = args[1]
    order3 = args[2]
    if len(args) >= 4:
        order4 = args[3]
    else:
        order4 = 3

    layer.permute_param(order1, order2, order3, order4)
    log.cnet.add_layer(layer)
    return x


#contiguous
def _contiguous(input, *args):
    x = raw__contiguous__(input, *args)
    name = log.add_layer(name='contiguous')
    log.add_blobs([x], name='contiguous_blob')
    layer = caffe_net.Layer_param(name=name,
                                  type='NeedRemove',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])
    log.cnet.add_layer(layer)
    return x


#pow
def _pow(input, *args):
    x = raw__pow__(input, *args)
    log.add_blobs([x], name='pow_blob')
    return x


#sum
def _sum(input, *args):
    x = raw__sum__(input, *args)
    log.add_blobs([x], name='sum_blob')
    return x


# sqrt
def _sqrt(input, *args):
    x = raw__sqrt__(input, *args)
    log.add_blobs([x], name='sqrt_blob')
    return x


# unsqueeze
def _unsqueeze(input, *args):
    x = raw__unsqueeze__(input, *args)
    log.add_blobs([x], name='unsqueeze_blob')
    return x


def _expand_as(input, *args):
    # only support expand A(1, 1, H, W) to B(1, C, H, W)

    x = raw__expand_as__(input, *args)
    layer_name = log.add_layer(name="expand_as", with_num=True)
    log.add_blobs([x], name='expand_as_blob')
    layer = caffe_net.Layer_param(name=layer_name,
                                  type='Convolution',
                                  bottom=[log.blobs(input)],
                                  top=[log.blobs(x)])

    def constant_weight(shape):
        weights = np.ones(shape, dtype='float32')
        return weights

    channels = args[0].size(1)
    weight = constant_weight([channels, 1, 1, 1])
    layer.conv_param(channels,
                     kernel_size=1,
                     bias_term=False,
                     weight_filler_type='xavier')
    layer.add_data(weight)
    log.cnet.add_layer(layer)
    return x


# 核心组件，通过该类，实现对torch的function中的operators的输入，输出以及参数的读取
class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        # replace the raw function to replace function
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        if not NET_INITTED:
            return self.raw(*args, **kwargs)
        for stack in traceback.walk_stack(None):
            if 'self' in stack[0].f_locals:
                layer = stack[0].f_locals['self']
                if layer in layer_names:
                    log.pytorch_layer_name = layer_names[layer]
                    print(layer_names[layer])
                    break
        out = self.obj(self.raw, *args, **kwargs)
        # if isinstance(out,Variable):
        #     out=[out]
        return out


F.conv2d = Rp(F.conv2d, _conv2d)
F.linear = Rp(F.linear, _linear)
F.relu = Rp(F.relu, _relu)
F.leaky_relu = Rp(F.leaky_relu, _leaky_relu)
F.max_pool2d = Rp(F.max_pool2d, _max_pool2d)
F.avg_pool2d = Rp(F.avg_pool2d, _avg_pool2d)
F.adaptive_avg_pool2d = Rp(F.adaptive_avg_pool2d, _adaptive_avg_pool2d)
F.dropout = Rp(F.dropout, _dropout)
F.threshold = Rp(F.threshold, _threshold)
F.prelu = Rp(F.prelu, _prelu)
F.batch_norm = Rp(F.batch_norm, _batch_norm)
F.instance_norm = Rp(F.instance_norm, _instance_norm)
F.layer_norm = Rp(F.layer_norm, _layer_norm)
F.pad = Rp(F.pad, _ReflectionPad2d)
F.softmax = Rp(F.softmax, _softmax)
F.conv_transpose2d = Rp(F.conv_transpose2d, _conv_transpose2d)
F.interpolate = Rp(F.interpolate, _interpolate)
# F.sigmoid = Rp(F.sigmoid, _sigmoid)
F.tanh = Rp(F.tanh, _tanh)
F.tanh = Rp(F.tanh, _tanh)
F.hardtanh = Rp(F.hardtanh, _hardtanh)
F.silu = Rp(F.silu, _silu)
# F.l2norm = Rp(F.l2norm,_l2Norm)
torch.sigmoid = Rp(torch.sigmoid, _sigmoid)
torch.split = Rp(torch.split, _split)
torch.max = Rp(torch.max, _max)
torch.cat = Rp(torch.cat, _cat)
torch.div = Rp(torch.div, _div)
torch.flatten = Rp(torch.flatten, _flatten)
torch.bmm = Rp(torch.bmm, _bmm)
# torch.add = Rp(torch.add, _add)

# TODO: other types of the view function
try:
    raw_view = Variable.view
    Variable.view = _view
    raw_mean = Variable.mean
    Variable.mean = _mean
    raw__add__ = Variable.__add__
    Variable.__add__ = _add
    raw__iadd__ = Variable.__iadd__
    Variable.__iadd__ = _iadd
    raw__sub__ = Variable.__sub__
    Variable.__sub__ = _sub
    raw__isub__ = Variable.__isub__
    Variable.__isub__ = _isub
    raw__mul__ = Variable.__mul__
    Variable.__mul__ = _mul
    raw__imul__ = Variable.__imul__
    Variable.__imul__ = _imul
except:
    # for new version 0.4.0 and later version
    for t in [torch.Tensor]:
        raw_view = t.view
        t.view = _view
        raw_mean = t.mean
        t.mean = _mean
        raw_squeeze = t.squeeze
        t.squeeze = _squeeze
        raw__add__ = t.__add__
        t.__add__ = _add
        raw__iadd__ = t.__iadd__
        t.__iadd__ = _iadd
        raw__sub__ = t.__sub__
        t.__sub__ = _sub
        raw__isub__ = t.__isub__
        t.__isub__ = _isub
        raw__mul__ = t.__mul__
        t.__mul__ = _mul
        raw__imul__ = t.__imul__
        t.__imul__ = _imul
        raw__permute__ = t.permute
        t.permute = _permute
        raw__contiguous__ = t.contiguous
        t.contiguous = _contiguous
        raw__pow__ = t.pow
        t.pow = _pow
        raw__sum__ = t.sum
        t.sum = _sum
        raw__sqrt__ = t.sqrt
        t.sqrt = _sqrt
        raw__unsqueeze__ = t.unsqueeze
        t.unsqueeze = _unsqueeze
        raw__expand_as__ = t.expand_as
        t.expand_as = _expand_as


# 清楚缓存的函数
def clear_all():
    log.clear()
    layer_names.clear()

# 转换模型的主要函数
def trans_net(net, input_var, name='TransferedPytorchModel'):
    print('Starting Transform, This will take a while')
    log.init([input_var])
    log.cnet.net.name = name
    log.cnet.net.input.extend([log.blobs(input_var)])
    log.cnet.net.input_dim.extend(input_var.size())
    global NET_INITTED
    NET_INITTED = True
    for name, layer in net.named_modules():
        layer_names[layer] = name
    print("torch ops name:", layer_names)
    out = net.forward(input_var)
    print('Transform Completed')
    return out


def save_prototxt(save_name):
    log.cnet.remove_layer_by_type("NeedRemove")
    log.cnet.save_prototxt(save_name)


def save_caffemodel(save_name):
    log.cnet.save(save_name)
