package com.skt.spark.r2.ml

import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import scala.collection.immutable
import scala.collection.mutable

object FlashBaseMLPipeline {

  import DateUtil._

  var spark: SparkSession = _
  var sparkContext: SparkContext = _

  def main(args: Array[String]) {
    val tfNetPath = args(0)
    val fbHost = args(1)
    val fbPort = args(2).toInt
    val batchSize = args(3).toInt

    spark = SparkSession.builder().appName("fb-ml-pipeline").getOrCreate()
    sparkContext = NNContext.initNNContext()

    val initialTime = "20190920094500"
    val normR2Df = normalizedDF(createR2DataFrame(fbHost, fbPort))

    var (start, end) = (0L, 0L)
    start = System.nanoTime()
    val cachedInputQueue = cacheInputFeaturesQueue(normR2Df, initialTime)
    end = System.nanoTime()
    println(s"The time of building Input-Queue is ${(end - start) / 1.0e9}s")

    start = System.nanoTime()
    val cachedMemoryQueue = cacheMemoryFeaturesQueue(normR2Df, beforeDays(afterMinutes(initialTime, 5), 1))
    end = System.nanoTime()
    println(s"The time of building Memory-Queue is ${(end - start) / 1.0e9}s")

    start = System.nanoTime()
    buildTensorRDD(
      buildInputFeatures(normR2Df, afterMinutes(initialTime, 5), cachedInputQueue),
      buildMemoryFeatures(normR2Df, beforeDays(afterMinutes(initialTime, 10), 1), cachedMemoryQueue),
      batchSize)
      .collect()
    end = System.nanoTime()
    println(s"The time of pre-processing is ${(end - start) / 1.0e9}s")

    start = System.nanoTime()
    val btfnet = ModelBroadcast[Float]().broadcast(sparkContext, TFNet(tfNetPath))
    end = System.nanoTime()
    println(s"The time of broadcasting model is ${(end - start) / 1.0e9}s")

    start = System.nanoTime()
    buildInferenceRDD(
      buildTensorRDD(
        buildInputFeatures(normR2Df, afterMinutes(initialTime, 5), cachedInputQueue),
        buildMemoryFeatures(normR2Df, beforeDays(afterMinutes(initialTime, 10), 1), cachedMemoryQueue),
        batchSize),
      btfnet
    ).collect()
    end = System.nanoTime()
    println(s"Total time of pre-processing and inference is ${(end - start) / 1.0e9}s")
  }

  /**
    * Cache a Map of ('cellID' -> 'buffer of recent 50 min features sorted by evt_dtm')
    * into each partitions.
    * Features are partitioned by unique CellId and distributed into Spark partitions.
    */
  def cacheInputFeaturesQueue(
      normR2Df: DataFrame,
      time: String)
  : RDD[immutable.Map[Int, mutable.ArrayBuffer[Array[Float]]]] = {
    val filter = buildTimeFilter("evt_dtm", time, 45, 1)
    val cachedInput = normR2Df.where(filter).rdd.map(row => (row.getInt(1), row))
      .aggregateByKey(new mutable.ArrayBuffer[(String, Array[Float])]())(
        (buf, row) => {
          val features = (for (i <- 2 until row.size) yield row.getFloat(i)).toArray
          buf.append((row.getString(0), features))
          buf
        }, (b1, b2) => b1 ++= b2)
      .mapPartitions(it => Iterator.single(it.map {
        case (cellId, featureBuf) => (cellId, featureBuf.sortBy(_._1).map(_._2))
      }.toMap),
        preservesPartitioning = true)
      .cache()
    cachedInput.count()

    cachedInput
  }

  /**
    * Build a inputRDD(rdd of a Map of ('cellID' -> 'input Features')).
    * It is built by concatenating new 5 min features with last 45 minutes features of cached
    * input-features in each partitions.
    */
  def buildInputFeatures(
      normR2Df: DataFrame,
      time: String,
      cachedInputFeatures: RDD[Map[Int, mutable.ArrayBuffer[Array[Float]]]])
  : RDD[immutable.Map[Int, mutable.ArrayBuffer[Array[Float]]]] = {
    val filter = buildTimeFilter("evt_dtm", time, 0, 1)
    normR2Df.where(filter).rdd
      .map(row => (row.getInt(1), row))
      .partitionBy(cachedInputFeatures.partitioner.get)
      .zipPartitions(cachedInputFeatures, preservesPartitioning = true) {
        (newRowIter, cached) =>
          if (newRowIter.isEmpty) {
            cached
          } else {
            val cachedMap = cached.next()
            val newMap = new mutable.HashMap[Int, mutable.ArrayBuffer[Array[Float]]]()
            newRowIter.foreach {
              newRow =>
                val buffer = cachedMap(newRow._1)
                val features = (for (i <- 2 until newRow._2.size) yield newRow._2.getFloat(i)).toArray
                newMap.put(newRow._1, buffer.drop(1) += features)
            }
            Iterator.single(newMap.toMap)
          }
      }
  }

  /**
    * Cache a Map of ('cellID' -> '7 days x buffer of 55 min features sorted by evt_dtm') into each
    * partitions.
    * Features are partitioned by unique CellId and distributed into Spark partitions.
    */
  def cacheMemoryFeaturesQueue(
      normR2Df: DataFrame,
      time: String)
  : RDD[immutable.Map[Int, Array[mutable.ArrayBuffer[Array[Float]]]]] = {
    val filter = buildTimeFilter("evt_dtm", time, 50, 7)
    val cachedMemory = normR2Df.where(filter).rdd.map(row => (row.getInt(1), row))
      .aggregateByKey(new mutable.ArrayBuffer[(String, Array[Float])]())(
        (buf, row) => {
          val features = (for (i <- 2 until row.size) yield row.getFloat(i)).toArray
          buf.append((row.getString(0), features)) // Tuple of (DT, Features)
          buf
        }, (b1, b2) => b1 ++= b2)
      .mapPartitions({
        it: Iterator[(Int, mutable.ArrayBuffer[(String, Array[Float])])] =>
          Iterator.single[Map[Int, Array[mutable.ArrayBuffer[Array[Float]]]]](
            it.map {
              case (cellId, fs) => (cellId, fs.sortBy(_._1).map(_._2).grouped(11).toArray)
            }.toMap)
      }, preservesPartitioning = true)
      .cache()

    cachedMemory.count()
    cachedMemory
  }

  /**
    * Build a memoryRDD(rdd of a Map of ('cellID' -> 'memory features')).
    * It is built by concatenating new 7days x 5 min features with 7days x last 50 minutes features
    * of cached memory feature in each partitions.
    */
  def buildMemoryFeatures(
      normR2Df: DataFrame,
      time: String,
      cachedMemory: RDD[immutable.Map[Int, Array[mutable.ArrayBuffer[Array[Float]]]]])
  : RDD[immutable.Map[Int, Array[mutable.ArrayBuffer[Array[Float]]]]] = {
    val filter = buildTimeFilter("evt_dtm", time, 0, 7)
    normR2Df.where(filter).rdd
      .map(row => (row.getInt(1), row))
      .partitionBy(cachedMemory.partitioner.get)
      .zipPartitions(cachedMemory, preservesPartitioning = true) {
        (newRowIter, cachedMapIter) =>
          if (newRowIter.isEmpty) {
            cachedMapIter
          } else {
            val cachedMap = cachedMapIter.next()
            val newMemoryMap = new mutable.HashMap[Int, Array[mutable.ArrayBuffer[Array[Float]]]]()
            newRowIter.toArray.groupBy(_._1).values.iterator.foreach {
              newRows: Array[(Int, Row)] =>
                val cellId = newRows(0)._1
                val memories = cachedMap(cellId)
                val newFeatures = newRows.map(_._2).sortBy(r => r.getString(0)).map {
                  row => (for (i <- 2 until row.size) yield row.getFloat(i)).toArray
                }
                val newMemory = Array.ofDim[mutable.ArrayBuffer[Array[Float]]](memories.length)
                for (i <- memories.indices) {
                  newMemory(i) = memories(i).drop(1)
                  newMemory(i) += newFeatures(i)
                }
                newMemoryMap.put(cellId, newMemory)
            }
            Iterator.single(newMemoryMap.toMap)
          }
      }
  }

  /**
    * Build TensorRDD with inputRDD and memoryRDD.
    * Input features and memory features are zipped and grouped with batchSize in each partitions.
    */
  def buildTensorRDD(
      inputRDD: RDD[immutable.Map[Int, mutable.ArrayBuffer[Array[Float]]]],
      memoryRDD: RDD[immutable.Map[Int, Array[mutable.ArrayBuffer[Array[Float]]]]],
      batchSize: Int
  ): RDD[Table] = {
    inputRDD.zipPartitions(memoryRDD, preservesPartitioning = true) {
      (iterInput, iterMemory) =>
        val inputMap = iterInput.next()
        val memoryMap = iterMemory.next()
        inputMap.iterator.map {
          case (cellId: Int, inputFs: mutable.ArrayBuffer[Array[Float]]) =>
            val memoryFs = memoryMap(cellId)
            (inputFs.flatten.toArray, memoryFs.flatMap(fs => fs.flatten))
        }.grouped(batchSize).map {
          tensors: Seq[(Array[Float], Array[Float])] =>
            val x = tensors.flatMap(_._1).toArray
            val m = tensors.flatMap(_._2).toArray
            val inputXData = Tensor[Float](x, Array(x.length / 10 / 8, 10, 8))
            val inputMData = Tensor[Float](m, Array(m.length / 77 / 8, 77, 8))
            T(inputXData, inputMData)
        }
    }
  }

  /**
    * Create inferenceRDD which applies tfModel to the tensorRDD.
    */
  def buildInferenceRDD(tensorRDD: RDD[Table], btfnet: ModelBroadcast[Float]): RDD[Activity] = {
    tensorRDD.mapPartitions(
      iterTable => {
        iterTable.toArray.map(btfnet.value().forward).toIterator
      }, preservesPartitioning = true)
  }

  /**
    * Create DataFrame which has FrashBase as data source.
    */
  def createR2DataFrame(fbHost: String, fbPort: Int): DataFrame = {
    val params = Map("table" -> "1",
      "host" -> fbHost,
      "port" -> fbPort.toString,
      "partitions" -> "evt_dtm random",
      "mode" -> "nvkvs",
      "group_query_enabled" -> "no",
      "group_size" -> "44",
      "query_result_partition_cnt_limit" -> "400000000",
      "query_result_task_row_cnt_limit" -> "10000000",
      "query_result_total_row_cnt_limit" -> "2147483647",
      "at_least_one_partition_enabled" -> "no")

    val fields = "evt_dtm,uniq_id,rsrp,rsrq,dl_prb_usage_rate,sinr,ue_tx_power,phr,ue_conn_tot_cnt,cqi,random"
      .split(',')
      .map {
        case "evt_dtm" => StructField("evt_dtm", StringType)
        case name@("uniq_id" | "random") => StructField(name, IntegerType)
        case fieldName => StructField(fieldName, FloatType)
      }

    val schema = StructType(fields)
    val r2Df = spark.read.format("r2")
      .options(params)
      .schema(schema)
      .load()
      .drop("random")
    r2Df
  }

  /**
    * Build DataFrame which applies normalization to each features.
    */
  def normalizedDF(dfToNorm: DataFrame): DataFrame = {
    val minMaxMap = Map[String, (Float, Float)](
      "rsrp" -> (-121.0f, 0.0f),
      "rsrq" -> (-20.0f, 0.0f),
      "dl_prb_usage_rate" -> (1.0f, 99.57666778564453f),
      "sinr" -> (-3.676666736602783f, 20.5f),
      "ue_tx_power" -> (-10.943333625793457f, 23.0f),
      "phr" -> (0.5f, 52.91666793823242f),
      "ue_conn_tot_cnt" -> (0.0f, 144.63333129882812f),
      "cqi" -> (1.9620689153671265f, 14.984615325927734f)
    )

    var df = dfToNorm
    for (col <- minMaxMap) {
      val colName = col._1
      val (min, max) = col._2
      df = df.withColumn(colName, (((df(colName) - min) * 2.0f / (max - min)) + -1.0f).cast(FloatType))
    }
    df
  }
}
