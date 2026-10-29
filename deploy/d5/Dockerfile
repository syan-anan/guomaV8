FROM python:3.10-slim
WORKDIR /app
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.server.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn --default-timeout=120 -r requirements.server.txt
COPY run.py config.py ./
COPY api ./api
COPY solver ./solver
EXPOSE 15666
CMD ["python","run.py"]
