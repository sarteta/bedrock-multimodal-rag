FROM python:3.13-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --prefix=/install --no-deps . \
 && pip install --prefix=/install \
        "boto3>=1.34.0" \
        "psycopg[binary]>=3.1.0" \
        "pgvector>=0.2.4" \
        "pydantic>=2.5.0" \
        "tenacity>=8.2.0" \
        "click>=8.1.0" \
        "rank-bm25>=0.2.2"


FROM python:3.13-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/sarteta/bedrock-multimodal-rag"
LABEL org.opencontainers.image.description="Multimodal RAG on AWS Bedrock: text + image retrieval with pgvector and a real evaluator"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 app \
 && useradd  --system --uid 10001 --gid app --create-home app

COPY --from=builder /install /usr/local

USER app
WORKDIR /home/app

# Image ships the bedrock_rag library + deps. Use `docker run -it ... python` and
# import the modules you need (retrieval, eval, bedrock_client). Requires AWS
# creds and DATABASE_URL at runtime, e.g.:
#   docker run --rm -e DATABASE_URL=postgresql://... \
#     -v $HOME/.aws:/home/app/.aws:ro \
#     ghcr.io/sarteta/bedrock-multimodal-rag \
#     -c "from bedrock_rag import retrieval; print(retrieval.__name__)"
ENTRYPOINT ["python"]
CMD ["-c", "import bedrock_rag, bedrock_rag.retrieval, bedrock_rag.eval, bedrock_rag.bedrock_client; print('bedrock_rag ready')"]
