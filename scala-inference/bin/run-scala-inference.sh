#!/bin/bash

if [ -z "${ANALYTICS_ZOO_HOME}" ]; then
    echo "please first download analytics zoo and set ANALYTICS_ZOO_HOME"
    exit 1
fi

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-scala-with-zoo.sh --driver-memory 20g --class Main target/mem-inference-1.0-SNAPSHOT-jar-with-dependencies.jar ../tfnet ../data/test_x.npy ../data/test_m.npy 65536 false

