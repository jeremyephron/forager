#!/usr/bin/env bash

export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcloud-credentials.json"
python3 manage.py makemigrations && python3 manage.py migrate
python3 manage.py runserver
