FROM python:3.8.15


# 1、复制相应的文件到tmp下
WORKDIR /workspace
COPY . ./

# 2、安装依赖文件
RUN pip3 install --upgrade pip && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements && \
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu && \
apt update && apt install -y --no-install-recommends openssh-server && apt install -y --force-yes vim

