# 模型转换服务器开发(flask)


## 1. 目的

- 创建一个网页版的模型转换工具，转换代码都放入到flask的后端服务中，前端主要用于进行转换路线选择和文件上传

## 2. 特点

- 前端采用分片方式上传大文件给后端，后端接收全部分片后，将其组合成一个文件  
- 支持多用户同时上传，互不干扰  
- 支持多种前端训练模型之间的转换，目前完善了pytorch2caffe的接口
- 对于用户上传的python文件，实现了动态加载和执行(很容易被用户的python文件下毒，需要注意)

## 3. 容器部署说明

- 获取相关容器（也可以用保存下的python_3.8.15.tar）
  ```bash
  docker pull python:3.8.15
  ```
- dockerfile构建部署镜像
  ```bash
  docker build -t flask_modeltrans:1.0 ./
  ```
- 开启容器
  ```bash
  docker run -it -p 4001:80 -p 4002:22 --privileged --net=bridge --ipc=host --pid=host flask_modeltrans:1.0 /bin/bash
  ```
  - 注意点：
  ```
  这时候已经进入了容器，需要注意：（1）设置root密码才能vscode远程；（2）关掉/etc/ssh/sshd_config中的密码设置方可vscode远程；（3）文件保存至worspace目录，即远程目录。
  ```
- 后端服务开启（直接跳到5）

## 4. 安装相关依赖(主机部署,容器不需要)

1. 安装python相关依赖库
    ```bash
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements
    ```

2. 安装lsof端口查看工具（可选）
    ```bash
    yum install lsof
    ```

## 5. 后端服务开启

  需要开启三个终端启动该服务,依次开启下面的相关命令(使用vscode即可)

- 终端一:
  ```bash
  chmod +x run-redis.sh
  ./run-redis.sh
  ```
- 终端二:
  ```bash
  celery -A server.celery worker --loglevel=debug --concurrency=8 -P threads -c 1000 -E
  ```
- 终端三:
  ```bash
  python3 server.py
  ```

## 6. 用户访问方法

- 情况一：该服务部署在主机上，那么服务的默认访问网址是http://主机ip:xxxx
- 情况二：该服务部署在主机的容器上，假设容器上的port映射为yyyy:80，那么服务的默认访问网址是http://主机ip:yyyy

## 7. 注意事项

详情请见[模型转换工具开发.docx](模型转换工具开发.docx)





  
