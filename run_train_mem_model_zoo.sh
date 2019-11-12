
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master local[4] \
 --driver-memory 20g \
 train_mem_model_zoo.py /home/jwang/git/ARMemNet-BigDL_jennie/data/aggregated_5min_scaled.csv 2700 1000
