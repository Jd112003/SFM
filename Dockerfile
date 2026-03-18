FROM colmap/colmap:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

COPY pyproject.toml README.md config.example.yaml ./
COPY sfm_pipeline ./sfm_pipeline

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "sfm_pipeline"]
