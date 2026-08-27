# 多阶段构建：先安装依赖，再运行服务
FROM docker.1ms.run/library/python:3.10-slim

WORKDIR /app

# 安装系统依赖（OpenCV 需要），使用阿里云源加速
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 暴露端口
EXPOSE 15666

# 启动服务
CMD ["python", "run.py"]
