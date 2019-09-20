#!/bin/bash

if [ -z "${ANALYTICS_ZOO_HOME}" ]; then
    echo "please first download analytics zoo and set ANALYTICS_ZOO_HOME"
    exit 1
fi

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh $dir_path/../../export_for_scala.py



