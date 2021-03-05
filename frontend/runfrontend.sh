#!/usr/bin/env bash
export REACT_APP_SERVER_URL="http://35.199.179.109:8000"
export PORT=4000

npm run build
serve -s build
