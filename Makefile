init: config.json
	mkdir -p data/submissions logs

venv:
	python3 -mvenv ~/venvs/porto

packages:
	pip install -r requirements.txt

config.json:
	cp config.json.sample config.json

clean:
	rm data/submissions/*
