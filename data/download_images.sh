#!/bin/bash

DATASETS_URLS=$1
DST_DIR=$2
DATASETS=()
while IFS= read -r line || [[ "$line" ]]; do
  DATASETS+=("$line")
done < $DATASETS_URLS

for dataset_url in ${DATASETS[@]}; do
  echo $dataset_url
  kaggle datasets download -p $DST_DIR --unzip $dataset_url
done
