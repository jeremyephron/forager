#!/usr/bin/env bash


#export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcloud-credentials.json"
# source ./cseInfo
python3 manage.py makemigrations && python3 manage.py migrate
#python3 manage.py runsslserver
uvicorn --host localhost --port 8000 forager_server.asgi:application
