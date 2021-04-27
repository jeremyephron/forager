#!/usr/bin/env bash
ip=$(curl -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip
)
export REACT_APP_SERVER_URL="http://${ip}:8000"
export PORT=4000

npm run start
