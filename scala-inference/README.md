## Steps to run scala inference code

You will need Spark, Analytics-Zoo and TensorFlow as described in `inference_mem_model_zoo_readme.md`

1. go to directory scala-inference and run `mvn package` and it will generate a few jar files the `target` directory. 

2. run `bash bin/run-python-export.sh` to export the tensorflow model to a directory named tfnet under the project root
directory and export the preprocessed data into the data directory.

3. run `bash bin/run-scala-inference.sh` to run inference benchmark in scala.