FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install dependencies.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Run the web service on container startup.
COPY . ./
CMD exec uvicorn --host "0.0.0.0" --port $PORT --workers 1 --log-level warning handler:mapper
