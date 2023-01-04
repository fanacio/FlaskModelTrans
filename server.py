#!/usr/bin/env python3
# coding=utf-8

import os
from UserManage.user_info import user_manage
from flask import Flask, session, jsonify,request, Response, url_for, redirect, render_template as rt
from celery import Celery
import random
import time

app = Flask(__name__)
#设置session的会话密钥
app.config["SECRET_KEY"] = 'TPmi4aLWRbyVq8zu9v82dWYW1'

# 设置文件上传的目标文件夹
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 设置celery接口配置
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# 初始化Celery接口
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
# 初始化下载路径
basedir = os.path.abspath(os.path.dirname(__file__))  # 获取当前项目的绝对路径
file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])  # 拼接成合法文件夹地址
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
# 删除upload下的所有文件
import shutil
shutil.rmtree(file_dir)
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
# 进行人员管理
UsrInfo = user_manage()

@app.route('/', methods=['GET', 'POST'])
def index():
    # 保证客户端进来的时候能够保证存在会话
    usrid = session.get("usrid")
    if usrid is None or not UsrInfo.in_usr(usrid):
        usrid = UsrInfo.add_new()
        session["usrid"] = usrid
    if request.method == 'GET':
        return rt('./index.html')
    else:
        return redirect(url_for('index'))
# 模型转换的接口
@app.route("/ModelTrans", methods=["POST", "GET"])
def ModelTrans():
    org = request.form.get("org")
    target = request.form.get("after")
    return jsonify({"org":org, "after":target})


# 上传分块
@app.route('/file/upload', methods=['POST'])
def upload_part():  # 接收前端上传的一个分片
    # 保证客户端进来的时候能够保证存在会话
    usrid = session.get("usrid")
    if usrid is None or not UsrInfo.in_usr(usrid):
        usrid = UsrInfo.add_new()
        session["usrid"] = usrid
    if not os.path.exists(os.path.join(file_dir, str(usrid))):
        os.mkdir(os.path.join(file_dir, str(usrid)))
    task = request.form.get('task_id')  # 获取文件的唯一标识符
    chunk = request.form.get('chunk', 0)  # 获取该分片在所有分片中的序号
    filename = '%s%s%s' % (task, chunk, ".chunk")  # 构造该分片的唯一标识符

    upload_file = request.files['file']
    
    upload_file.save(os.path.join(os.path.join(file_dir, str(usrid)), filename))  # 保存分片到本地
    return rt('./index.html')


# 对分块进行合并
@app.route('/file/merge', methods=['GET'])
def upload_success():  # 按序读出分片内容，并写入新文件
    # 保证客户端进来的时候能够保证存在会话
    usrid = session.get("usrid")
    if usrid is None or not UsrInfo.in_usr(usrid):
        usrid = UsrInfo.add_new()
        session["usrid"] = usrid

    target_filename = request.args.get('filename')  # 获取上传文件的文件名
    task = request.args.get('task_id')  # 获取文件的唯一标识符

    # 创建本次传输任务的文件夹
    target_path = os.path.join(file_dir, str(usrid), str(task))
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    chunk = 0  # 分片序号
    target_filename = os.path.join(target_path, target_filename)
    
    with open(target_filename, 'wb') as target_file:  # 创建新文件
        while True:
            try:
                filename = '%s%d%s' % (task, chunk, ".chunk")
                filename = os.path.join(os.path.join(file_dir, str(usrid)), filename)
                source_file = open(filename, 'rb')  # 按序打开每个分片
                target_file.write(source_file.read())  # 读取分片内容写入新文件
                source_file.close()
            except IOError as msg:
                break
            chunk += 1
            os.remove(filename)  # 删除该分片，节约空间
    # 因为add_usr_file的时候，会检查是否存在target_filename，所以必须是在merge完成后，才进行add_usr_file操作
    UsrInfo.add_usr_file(usrid, target_filename)
    return rt('./index.html')


def all_files_path(rootDir):
    filepaths = []
    for root, dirs, files in os.walk(rootDir):     # 分别代表根目录、文件夹、文件
        for file in files:                         # 遍历文件
            file_path = os.path.join(root, file)   # 获取文件绝对路径
            filepaths.append(file_path)            # 将文件路径添加进列表
        for dir in dirs:                           # 遍历目录下的子目录
            dir_path = os.path.join(root, dir)     # 获取子目录路径
            all_files_path(dir_path)
    return filepaths

# 罗列列表的api，主要是进行下载功能
@app.route('/file/list', methods=['GET'])
def file_list():
    # 保证客户端进来的时候能够保证存在会话
    usrid = session.get("usrid")
    if usrid is None or not UsrInfo.in_usr(usrid):
        usrid = UsrInfo.add_new()
        session["usrid"] = usrid

    files = os.listdir(os.path.join(file_dir, str(usrid)))  # 获取文件目录
    files = map(lambda x: x if isinstance(x, str) else x.decode('utf-8'), files)  # 注意编码
    # if len(files) == 0:
    #     files = ["None"]
    return rt('./list.html', files=files)


# 无显示功能，纯粹作为一个后端api进行下载和传输
@app.route('/file/download/<filename>', methods=['GET'])
def file_download(filename):
    # 保证客户端进来的时候能够保证存在会话
    usrid = session.get("usrid")
    if usrid is None or not UsrInfo.in_usr(usrid):
        usrid = UsrInfo.add_new()
        session["usrid"] = usrid

    def send_chunk(usrid):  # 流式读取
        store_path = os.path.join(os.path.join(file_dir, usrid), filename)
        with open(store_path, 'rb') as target_file:
            while True:
                chunk = target_file.read(20 * 1024 * 1024)
                if not chunk:
                    break
                yield chunk

    return Response(send_chunk(usrid), content_type='application/octet-stream')

# 增加长时间任务运行接口
# 未来的模型转换接口就在此处
from ModelTrans.choose import check_enable
from Tools.unzip_zip import unzip, zipDir
@celery.task(bind=True)
def long_task(self, org='', target='', file=''):
    """这个就是长时间的任务"""
    # verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    # adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    # noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    # message = ''
    # total = random.randint(10, 50)
    # for i in range(total):
    #     if not message or random.random() < 0.25:
    #         message = '{0} {1} {2}...'.format(random.choice(verb),
    #                                           random.choice(adjective),
    #                                           random.choice(noun))
    #     self.update_state(state='PROGRESS',
    #                       meta={'current': i, 'total': total,
    #                             'status': message})
    #     time.sleep(1)
    # 正式的模型转换
    # 转换路线确认、文件前处理
    mf, transfun, input_type, output_type = check_enable(org, target)
    print(org, target, file)
    if mf and transfun is not None and input_type is not None:
        if file is not None and isinstance(file, list):
            filenum = len(input_type)
            if len(file) >= filenum:
                total = 100
                self.update_state(state='PROGRESS',
                                meta={'current': 0, 'total': total,
                                        'status': 'begin model transform'})
                id_workspace = os.path.join(*file[-1].split('/')[:-2])
                id_workspace = os.path.join('/', id_workspace)
                #文件前处理
                allfile = file[-filenum:]
                sortfile = [''] * filenum
                for i in allfile:
                    for j in input_type:
                        if i.endswith(j) and os.path.isfile(i):
                            sortfile[input_type[j]] = i
                # 对zip文件进行解压
                # 获取模型的文件名
                modelname = ''
                for j in input_type:
                    if j == '.zip':
                        sortfile[input_type[j]] = unzip(sortfile[input_type[j]])
                    else:
                        modelname = os.path.splitext(os.path.split(sortfile[input_type[j]])[-1])[0]
                self.update_state(state='PROGRESS', meta={'current': 10, 'total': total,'status': 'begin model transform'})
                # 输出文件路径构建
                outputfile = [''] * len(output_type)
                for i in output_type:
                    if i == '.zip':
                        outputfile[output_type[i]] = os.path.join(id_workspace, 'weights')
                    else:
                        outputfile[output_type[i]] = os.path.join(id_workspace, '{0}{1}'.format(modelname, i))
                # 开始转换
                sortfile.extend(outputfile)
                '''
                传参顺序为:原始模型路径、目标模型路径
                '''
                print('{0}\n'.format(sortfile))
                transfun(*sortfile)
                # 对需要进行压缩的文件夹进行压缩成zip
                for i in output_type:
                    if i == '.zip':
                        zipDir(outputfile[output_type[i]], outputfile[output_type[i]] + '.zip')
                        outputfile[output_type[i]] = outputfile[output_type[i]] + '.zip'
                
                # 转换成功，把转换后的模型路径进行返回，并且跳转到list目录下
                self.update_state(state='PROGRESS',
                                meta={'current': total, 'total': total,
                                        'status': 'end model transform'})
                outputfile = [os.path.split(i)[-1] for i in outputfile]
                return {'current': 100, 'total': 100, 'status': 'Task completed!',
                    'result': 42, 'output':outputfile}
    # self.update_state(state='FAILURE',
    #                 meta={'current': 0, 'total': 100,
    #                         'status': 'model transform failure'})
    return {'current': 99, 'total': 100, 'status': 'Task failure!'}

@app.route('/longtask', methods=['POST'])
def longtask():
    usrid = session.get("usrid")
    if usrid is None or not UsrInfo.in_usr(usrid):
        # 此时表示该用户没有传入文件，就开始进行转换操作了
        return jsonify({}), 505, {'Location': None}
    org = request.form.get("org")
    target = request.form.get("after")
    if org is None or target is None:
        # 此时表示未传入正确的参数
        return jsonify({}), 404, {'Location': None}
    print({'org':org, 'target':target, 'file':UsrInfo.list_usr_file(usrid)})
    task = long_task.apply_async(kwargs={'org':org, 'target':target, 'file':UsrInfo.list_usr_file(usrid)})
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            # 此时就是完成了转换工作，需要传输下载路径
            response['result'] = task.info['result']
            if 'output' in task.info:
                response['output'] = task.info['output']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=80, host="0.0.0.0")
