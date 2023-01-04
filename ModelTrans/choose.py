import os
import re

def list2dict(l):
    d = {}
    for i, j in enumerate(l):
        d[j] = i
    return d


orgmodel = ['darknet', 'pytorch', 'onnx', 'json', 'tensorflow']
targetmodel = ['darknet', 'pytorch', 'onnx', 'json', 'tensorflow', 'caffe']
model_type = {'darknet':{'.weights':0,'.cfg':1}, 
            'pytorch':{'.pth':0, '.py':1},
            'onnx':{'.onnx':0}, 
            'json':{'.json':0, '.zip':1}, 
            'tensorflow':{'.pb':0}, 
            'caffe':{'.caffemodel':0, '.prototxt':1}}
orgmodel = list2dict(orgmodel)
targetmodel = list2dict(targetmodel)
enable = [[False] * len(targetmodel)] * len(orgmodel)
enable_fun = dict()

# 以下罗列相应的转换路线支持
def add_enable(input ,output, fun):
    if input in orgmodel and output in targetmodel:
        enable[orgmodel[input]][targetmodel[output]] = True
        tname = '{0}-{1}'.format(input, output)
        enable_fun[tname] = fun
        

# 返回对应的方法以及是否存在转换路线
def check_enable(input, output):
    tname = '{0}-{1}'.format(input, output)
    if input in orgmodel and output in targetmodel and tname in enable_fun:
        return enable[orgmodel[input]][targetmodel[output]],  enable_fun[tname], model_type[input], model_type[output]
    else:
        return False, None, None, None


# 进行罗列相关的转换路线支持
# 进行转换的demo
import time
from shutil import copyfile, copytree
modeldemo = '/home/ModelTrans2/ModelDemo'
def d2o(wefile, cfg, onnx):
    print(wefile, cfg, onnx)
    copyfile(os.path.join(modeldemo, 'demo.onnx'), onnx)
    return 'None'

def d2j(wefile, cfg, json, wdir):
    print(wefile, cfg, json, wdir)
    copyfile(os.path.join(modeldemo, 'demo.json'), json)
    copytree(os.path.join(modeldemo, 'demo'), wdir)
    return 'None'

def p2o(wefile, model_path, onnx):
    print(wefile, model_path, onnx)
    copyfile(os.path.join(modeldemo, 'demo.onnx'), onnx)
    return 'None'

def p2j(wefile, model_path, json, wdir):
    print(wefile, model_path, json, wdir)
    copyfile(os.path.join(modeldemo, 'demo.json'), json)
    copytree(os.path.join(modeldemo, 'demo'), wdir)
    return 'None'

# 该版本可行，能够传入单个py脚本是可行的
def p2c(wefile, model_path, caffef, modelf):
    print(wefile, model_path, caffef, modelf)
    import torch
    import importlib
    from .pytorch2caffe import pytorch2caffe
    # 这些尝试都失败了
    model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
    # model_path = model_path.split('/')[-3:-1]
    # model_p = []
    # model_p.extend(model_path)
    # model_p.append(model_name)
    # import sys
    # sys.path.append('../')
    # import upload
    # print('.'.join(model_p))
    # # 此时需要变成相对导入
    # model = importlib.import_module( '.'.join(model_p))
    # 接下来直接尝试把py文件复制过去，然后直接包含的操作
    usrid = model_path.split('/')[-3]
    new_name = '{0}.py'.format(usrid)
    new_path = []
    new_path.extend(model_path.split('/')[:-4])
    new_path.append('ModelTrans')
    new_path = os.path.join('/'.join(new_path), new_name)
    import sys
    print(sys.argv[0])
    copyfile(model_path, new_path)
    try:
        model = importlib.import_module( '.' + new_name.split('.')[0], package='ModelTrans')
        model = importlib.reload(model)
        input_data = torch.rand(model.shape, dtype=torch.float).cpu()
        net = model.model()
        print(type(net))
        net_dict = torch.load(wefile, map_location='cpu')
        net.load_state_dict(net_dict, False)
        net.eval().cpu()
        pytorch2caffe.trans_net(net, input_data, model_name)
        pytorch2caffe.save_prototxt(modelf)
        pytorch2caffe.save_caffemodel(caffef)
    finally:
        os.remove(new_path)
        pytorch2caffe.log = pytorch2caffe.TransLog()
        pytorch2caffe.layer_names = {}
        
        # del model
    

# 增加支持函数的传参如下所示
'''
原始模型，目标模型，转换函数，传入文件的类型dict={'.caffemodel':0, '.prototxt':1}
'''
add_enable('darknet', 'onnx', d2o)
add_enable('darknet', 'json', d2j)
add_enable('pytorch', 'onnx', p2o)
add_enable('pytorch', 'json', p2j)
add_enable('pytorch', 'caffe', p2c)