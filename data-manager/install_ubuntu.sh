#!/usr/bin/env bash
set -e

APP_DIR="${1:-/opt/data-manager}"

sudo mkdir -p "$APP_DIR"
sudo rsync -a --delete ./ "$APP_DIR/"

cd "$APP_DIR"

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

sudo mkdir -p data/db logs

# optional: env file
if [ -f env.example ] && [ ! -f .env ]; then
  cp env.example .env
  echo "Created .env from env.example (fill in API keys)"
fi

echo "Installed to $APP_DIR"
echo "Run collector: $APP_DIR/venv/bin/python $APP_DIR/run.py run"
echo "Run web:       $APP_DIR/venv/bin/python $APP_DIR/run.py web --host 0.0.0.0 --port 5050"

