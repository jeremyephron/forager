FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies.
COPY requirements-build.txt ./
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3.7-dev wget \
    && pip install -r requirements-build.txt \
    && pip install -r requirements.txt \
    && grep -F -x -v -f requirements.txt requirements-build.txt | pip uninstall -y -r /dev/stdin \
    && apt-get purge -y --auto-remove build-essential python3.7-dev

# Download external resources.
COPY resources.txt ./
RUN apt-get install wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget -i resources.txt \
    && apt-get purge -y --auto-remove wget

# Run the web service on container startup.
COPY . ./
ENV LRU_CACHE_CAPACITY=1
CMD exec uvicorn --host "0.0.0.0" --port $PORT --workers 1 --log-level warning handler:mapper
