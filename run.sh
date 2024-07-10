#!/bin/bash

PYTHON_VERSION="python3.10"

if ! command -v $PYTHON_VERSION &> /dev/null
then
    echo "$PYTHON_VERSION could not be found. Please install Python 3.10.14."
    exit 1
fi

$PYTHON_VERSION -m venv venv
source venv/bin/activate

pip install -r requirements.txt

if ! pip show nbconvert &> /dev/null
then
    pip install nbconvert
fi

cd examples/model_runners

for notebook in *.ipynb; do
    jupyter nbconvert --to notebook --execute "$notebook" --inplace
done

deactivate
