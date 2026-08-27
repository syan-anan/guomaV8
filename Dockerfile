# 验证码识别服务 Dockerfile（与服务器已验证版本一致）
FROM python:3.10-slim

WORKDIR /app

# 系统依赖：阿里云源 + OpenCV 运行库（含 libgl1）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（清华镜像加速 + 超时放宽）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --default-timeout=120 \
    -r requirements.txt

# 拷贝项目代码
COPY . .

# 暴露端口
EXPOSE 15666

# 启动服务
CMD ["python", "run.py"]
