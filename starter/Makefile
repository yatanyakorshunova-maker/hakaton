PYTHON ?= python3
ARGS ?=

.PHONY: build train submission

build:
	$(PYTHON) -m pip install --editable . --no-build-isolation

train:
	$(PYTHON) train.py $(ARGS)

submission:
	$(PYTHON) package_submission.py
