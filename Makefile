init:
    mkdir -p data/submissions logs
    pip install -r requirements.txt

config.json:
    cp config.json.sample config.json