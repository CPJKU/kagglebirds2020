#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Uploads the data in kaggle_upload/ as a new version of the"
    echo "janschl/kagglebirds-pytorch-data dataset on Kaggle."
    echo
    echo "Usage: $0 COMMENT"
    echo "    COMMENT: A string (<= 50 chars) describing the version."
fi
kaggle datasets version -m "$1" -p kaggle_upload/
