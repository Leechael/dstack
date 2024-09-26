#!/bin/bash
set -e
apt-get update
apt-get install wireguard-tools -y
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh --version 27.1.2
systemctl daemon-reload
systemctl enable app-compose.service
systemctl enable tappd.service
systemctl enable tboot.service
echo "port=4050" > /etc/tdx-attest.conf