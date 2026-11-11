.PHONY: setup data analyze test all

setup:
	python -m pip install -r requirements.txt

data:
	python data/generate_data.py --n 40000 --seed 42 --out data/raw

analyze:
	PYTHONPATH=src python -m custsegab.pipeline --data data/raw --out reports

test:
	python -m pytest -q

all: data analyze test
