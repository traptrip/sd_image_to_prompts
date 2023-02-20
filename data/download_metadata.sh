#!/bin/bash

DST_DIR=$1
kaggle datasets download -p $DST_DIR --unzip alexandreteles/diffusiondb-metadata
