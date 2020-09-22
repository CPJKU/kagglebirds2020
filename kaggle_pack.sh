#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 MODELFILE [MODELFILE ...]"
    echo "Creates kaggle_upload/kaggle_py.tgz and kaggle_upload/kaggle_mdl.tgz."
    echo "The former contains the required Python scripts, the latter the model"
    echo "weights."
    exit 1
fi

modelfiles="${@:1}"
tar -czf kaggle_upload/kaggle_mdl.tgz $modelfiles ${modelfiles//.mdl/.vars}
tar -czf kaggle_upload/kaggle_py.tgz predict_kagglebirds2020.py definitions/
