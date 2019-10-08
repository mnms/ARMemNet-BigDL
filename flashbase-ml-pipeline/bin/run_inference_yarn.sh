#!/bin/bash
if [ -z "${ANALYTICS_ZOO_HOME}" ]; then
    echo "please first download analytics zoo and set ANALYTICS_ZOO_HOME"
    exit 1
fi

FLASHBASE_LIB=''
if [ -z "${FLASHBASE_LIB}" ]; then
    echo "please set path of flashbase libraries"
    exit 1
fi


FLASHBASE_CLASSPATH=$(find $FLASHBASE_LIB -name 'tsr2*' -o -name 'spark-r2*' -o -name '*jedis*' -o -name 'commons*' -o -name 'jdeferred*' \
-o -name 'geospark*' -o -name 'gt-*' | tr '\n' ':')
TF_NET_PATH=''
DRIVER_MEMORY=20g
EXECUTOR_MEMORY=20g
EXECUTOR_CORES=10
EXECUTOR_INSTANCES=5
FLASHBASE_HOST=fbg02
FLASHBASE_PORT=18700
BATCH_SIZE=512

if [ -z "${TF_NET_PATH}" ]; then
    echo "please specify directory of tf-model."
    exit 1
fi

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

bash $dir_path/../../scala-inference/bin/spark-submit-scala-with-zoo.sh --master yarn --deploy-mode client \
--driver-memory $DRIVER_MEMORY --executor-memory $EXECUTOR_MEMORY --num-executors $EXECUTOR_INSTANCES \
--conf spark.executor.cores=$EXECUTOR_CORES --driver-class-path $FLASHBASE_CLASSPATH --conf spark.executor.extraClassPath=$FLASHBASE_CLASSPATH \
--class com.skt.spark.r2.ml.FlashBaseMLPipeline ../target/flashbase-ml-pipeline-1.0-SNAPSHOT.jar \
$TF_NET_PATH $FLASHBASE_HOST $FLASHBASE_PORT $BATCH_SIZE
