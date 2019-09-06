import java.nio.file.Paths

import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.jetbrains.bio.npy.NpyFile

object Main {


  def testForLocal(tfnetPath: String, testXPath: String, testMPath: String, batchSize: Int) = {

    val xPath: java.nio.file.Path = Paths.get(testXPath)

    val testX = NpyFile.read(xPath, Int.MaxValue)

    val mPath: java.nio.file.Path = Paths.get(testMPath)

    val testM = NpyFile.read(mPath, Int.MaxValue)

    val inputXData = Tensor[Float](testX.asFloatArray(), testX.getShape)
    val inputMData = Tensor[Float](testM.asFloatArray(), testM.getShape)

    val length = testM.getShape()(0)

    val tfnet = TFNet(tfnetPath, TFNet.SessionConfig(0, 0))

    val start = System.nanoTime()
    val outputData = Tensor[Float](length, 8)
    var i = 0
    while (i < Math.ceil(length / (1.0 * batchSize))) {
      val input = T(inputXData.narrow(1, batchSize * i + 1, Math.min(batchSize, length - batchSize * i)),
        inputMData.narrow(1, batchSize * i + 1, Math.min(batchSize, length - batchSize * i)))

      val result = tfnet.forward(input)
      outputData.narrow(1, batchSize * i + 1, Math.min(batchSize, length - batchSize * i)).copy(result.toTensor)
      i += 1
    }

    val end = System.nanoTime()
    println(s"time is ${(end - start)/1.0e9}s")

  }

  def testForSpark(tfnetPath: String, testXPath: String, testMPath: String, batchSize: Int) = {

    val sc = NNContext.initNNContext()

    val xPath: java.nio.file.Path = Paths.get(testXPath)

    val testX = NpyFile.read(xPath, Int.MaxValue)

    val mPath: java.nio.file.Path = Paths.get(testMPath)

    val testM = NpyFile.read(mPath, Int.MaxValue)

    val length = testM.getShape()(0)

    val tfnet = TFNet(tfnetPath)

    val btfnet = ModelBroadcast[Float]().broadcast(sc, tfnet)

    val batchedX = testX.asFloatArray().grouped(batchSize * 10 * 8).toSeq

    val batchedM = testM.asFloatArray().grouped(batchSize * 77 * 8).toSeq

    val localInput = batchedX.zip(batchedM).map { case (x, m) =>
      val inputXData = Tensor[Float](x, Array(x.length/10/8, 10, 8))
      val inputMData = Tensor[Float](m, Array(m.length/77/8, 77, 8))
      val input = T(inputXData, inputMData)
      input
    }

    val inputRdd = sc.parallelize(localInput, 4).cache()
    inputRdd.count()


    val modelRdd = inputRdd.mapPartitions { part =>
      val localTFNet = btfnet.value()
      Iterator.single(localTFNet)
    }.cache()

    modelRdd.count()

    val results = inputRdd.zipPartitions(modelRdd) { case (dataIter, modelIter) =>
      val model = modelIter.next()
      val result = dataIter.toArray.map(model.forward)
      result.toIterator
    }

    var start = System.nanoTime()
    results.collect()
    var end = System.nanoTime()
    println(s"time is ${(end - start)/1.0e9}s")
  }

  def main(args: Array[String]): Unit = {

    val tfnetPath = args(0)
    val testXPath = args(1)
    val testMPath = args(2)
    val batchSize = args(3).toInt
    val islocal = args(4).toBoolean

    println(s"tfnet path is ${tfnetPath}")
    println(s"test_x path is ${testXPath}")
    println(s"test_m path is ${testMPath}")
    println(s"batch size is ${batchSize}")

    if (islocal) {
      testForLocal(tfnetPath, testXPath, testMPath, batchSize)
    } else {
      testForSpark(tfnetPath, testXPath, testMPath, batchSize)
    }
  }
}