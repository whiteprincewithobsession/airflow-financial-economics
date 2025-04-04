FROM apache/airflow:2.10.5

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        openjdk-17-jre-headless \
        python3-dev \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    pyspark==3.4.0 \
    pandas==1.3.0 \
    matplotlib==3.5.0 \
    seaborn==0.11.0 \
    statsmodels==0.14.0 \
    jinja2==3.0.0 \
    pdfkit==0.5.0 \
    wkhtmltopdf==0.12.6