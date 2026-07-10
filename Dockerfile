# MCI World Model — 生产容器
# 多阶段构建: builder (编译依赖) → runtime (精简运行时)

# ── Stage 1: Builder ──
FROM python:3.13-slim AS builder

WORKDIR /build

# 编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# 安装核心依赖 (不含 torch 等可选依赖)
RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir . || \
    pip install --no-cache-dir numpy>=1.26.0 networkx>=3.0 scipy

# ── Stage 2: Runtime ──
FROM python:3.13-slim AS runtime

LABEL maintainer="MCI World Model Team"
LABEL description="Causal World Model Engine — Production"

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制源码
COPY src/ /app/src/
COPY pyproject.toml /app/

# 安装为可编辑包
RUN pip install --no-cache-dir -e .

# 安装 curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 非 root 用户运行
RUN useradd -m -u 1000 mci && chown -R mci:mci /app
USER mci

# 环境变量
ENV MCI_LOG_LEVEL=INFO
ENV MCI_LOG_FORMAT=json
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MCI_HOST=0.0.0.0
ENV MCI_PORT=8080

# 暴露 API 端口
EXPOSE 8080

# 健康检查 — 打 API /health 端点
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=5s \
    CMD curl -sf http://localhost:8080/health || exit 1

# 启动 API server
CMD ["python", "-m", "mci_world_model.server.app"]
