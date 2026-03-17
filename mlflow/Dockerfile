# TF 2.12 base already ships with tensorflow, numpy, and libgomp1 — no need to reinstall them
FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

# Install only the non-TF dependencies (much faster — skips the 600 MB TF download)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas \
    mlflow==2.2.2 \
    python-box==6.0.2 \
    pyYAML \
    tqdm \
    ensure==1.0.2 \
    joblib \
    types-PyYAML \
    scipy \
    Flask \
    Flask-Cors \
    gdown \
    dvc

COPY . /app
ENV PYTHONPATH=/app/src
RUN pip install --no-cache-dir -e .

CMD ["python3", "app.py"]