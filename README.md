# Porto Seguro Kaggle Problem #
See [here](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data) for details

### Setup ###
```
mkdir -p data/submissions logs
pip install -r requirements.txt
cp config.json.sample config.json
emacs config.json # edit any values you like
```
or setup with make:
```
make init
make venv
. ~/venvs/porto/bin/activate
make packages
```

There is a `make clean` task to delete any unwanted submission files.

### Contact ###

* Tim Hoolihan
  * [twitter](https://twitter.com/thoolihan)
  * [github](https://github.com/thoolihan)
