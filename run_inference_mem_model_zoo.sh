#!/bin/bash
if [ -z "${ANALYTICS_ZOO_HOME}" ]; then
    echo "please first download analytics zoo and set ANALYTICS_ZOO_HOME"
    exit 1
fi

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] inference_mem_model_zoo.py
