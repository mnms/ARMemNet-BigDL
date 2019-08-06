Instructions to run this sample code


1. Install Spark

   Please refer to Spark official guide to install Spark. https://spark.apache.org/. Version is 2.x is recommended.
   
2. Install Analytics Zoo and set ANALYTICS_ZOO_HOME environment variable

   Please refer Analytics Zoo Website
 
3. Install TensorFlow (only needed in driver), currently only TensorFlow 1.10.0 is tested and supported

4. run the starting script
   ```bash
   bash run_inference_mem_model_zoo.sh
   ```
   
   The script will save the inference results in ./ARMem/zoo_results as text file. To reproduce the
exact same result as test_mem_model, the PARALLELISM and BATCH_PER_THREAD constants should be set to
1 and 1022 respectively.