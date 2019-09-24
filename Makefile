.PHONY: clean install update-setup.py

ENV ?= $(shell pipenv --venv)
RUN = . $(ENV)/bin/activate &&

install:
	$(RUN) pipenv install --ignore-pipfile  # install from Pipfile.lock
	$(RUN) python -m spacy download en
	-mkdir logs

clean:
	rm -rf index.txt

update-setup.py:
	$(RUN) pipenv lock --pre
	pipenv-setup sync
	gsed -i 's|==.*|",|' setup.py
