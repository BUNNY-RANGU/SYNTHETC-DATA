# Synthetic Medical Data Generator - container image for the web app.
#
# Build and run locally:
#     docker build -t synthmed .
#     docker run -p 8501:8501 synthmed
#     open http://localhost:8501
#
# Torch is installed from the CPU wheel index BEFORE requirements.txt, so the
# image does not drag in the ~2.5 GB CUDA toolkit. The project was developed and
# benchmarked CPU-only, and every published number is a CPU number.
#
# NOT verified in this environment: `docker` is not installed on the machine
# this project was built on, so this file is shipped as a deployment target that
# has been reviewed, not one that has been run. If the first build fails, the
# output of `docker build` will say which line - it will be a wheel-index or
# version-resolution issue, not an application issue.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# opacus and ctgan resolve their torch dependency against the CPU build already
# installed here, so neither pulls the CUDA wheel.
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

# Code, the two raw UCI tables, and the published results the frozen-cohort mode
# renders. results/ is an input to the UI, not a build artefact.
COPY synth/ ./synth/
COPY ui/ ./ui/
COPY data/ ./data/
COPY results/ ./results/
COPY .streamlit/ ./.streamlit/
COPY app.py run_pipeline.py make_figures.py ./

# Streamlit as a non-root user. The container is read-only in practice: nothing
# in app.py writes. run_pipeline.py does write to results/, so if you want to
# re-run the sweep inside the container, add `-v $PWD/results:/app/results`.
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit's own liveness endpoint - the same URL tests/test_app.py polls.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read() == b'ok' else 1)"

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true"]
