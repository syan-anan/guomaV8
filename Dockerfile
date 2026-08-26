# 多阶段构建：先安装依赖，再运行服务
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
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
CMD ["python", "api/server.py"]
