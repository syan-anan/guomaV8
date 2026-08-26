# 部署指南

## 环境要求
- Docker >= 20.10
- docker-compose >= 1.29

## 快速部署

1. 复制环境变量示例并编辑：
```bash
cp .env.example .env
# 编辑 .env 设置端口、管理员密钥等
```

2. 执行一键部署脚本：
```bash
bash deploy.sh
```

3. 查看服务状态：
```bash
docker-compose logs -f
```

## 手动部署

```bash
# 构建镜像
docker-compose build

# 后台启动
docker-compose up -d

# 停止
docker-compose down
```

## 端口与配置

- 默认端口：`15666`
- 配置文件：`.env`
- 日志挂载：`./logs`
- 缓存挂载：`./__cache`

## 健康检查

```bash
curl http://localhost:15666/
```
