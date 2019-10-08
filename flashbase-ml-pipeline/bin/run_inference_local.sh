#!/bin/bash
if [ -z "${ANALYTICS_ZOO_HOME}" ]; then
    echo "please first download analytics zoo and set ANALYTICS_ZOO_HOME"
    exit 1
fi

TF_NET_PATH=""
DRIVER_MEMORY="40g"
FLASHBASE_HOST=fbg02
FLASHBASE_PORT=18700
BATCH_SIZE=512

if [ -z "${TF_NET_PATH}" ]; then
    echo "please specify directory of tf-model."
    exit 1
fi

bash ${ANALYTICS_ZOO_HOME}/bin/spark-submit-scala-with-zoo.sh --master "local[*]" \
--driver-memory $DRIVER_MEMORY --class com.skt.spark.r2.ml.FlashBaseMLPipeline target/flashbase-ml-pipeline-1.0-SNAPSHOT.jar \
$TF_NET_PATH $FLASHBASE_HOST $FLASHBASE_PORT $BATCH_SIZE
