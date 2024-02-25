// Databricks notebook source
//Part 1

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint
def computeSummand(x: Vector, y: Double, theta: Vector): Vector = {
  val prediction = theta dot x
  val error = prediction - y
  val multiplied = x.toArray.map(_ * error)
  Vectors.dense(multiplied)
}
def createLabeledPoint(features: Vector, label: Double): LabeledPoint = {
  LabeledPoint(label, features)
}
def main(args: Array[String]): Unit = {
  val theta = Vectors.dense(0.5, 0.3, -0.2) 
  val x1 = Vectors.dense(1.0, 2.0, 3.0) 
  val y1 = 7.0                        
  val summand1 = computeSummand(x1, y1, theta)
  println("Summand 1:")
  println(summand1)
  val labeledPoint1 = createLabeledPoint(x1, y1)
  println("LabeledPoint 1:")
  println(labeledPoint1)
  val x2 = Vectors.dense(2.0, 3.0, 4.0) 
  val y2 = 10.0                          
  val summand2 = computeSummand(x2, y2, theta)
  println("Summand 2:")
  println(summand2)
  val labeledPoint2 = createLabeledPoint(x2, y2)
  println("LabeledPoint 2:")
  println(labeledPoint2)
  val x3 = Vectors.dense(3.0, 4.0, 5.0) 
  val y3 = 13.0                          
  val summand3 = computeSummand(x3, y3, theta)
  println("Summand 3:")
  println(summand3)
  val labeledPoint3 = createLabeledPoint(x3, y3)
  println("LabeledPoint 3:")
  println(labeledPoint3)
  val x4 = Vectors.dense(4.0, 5.0, 6.0) 
  val y4 = 16.0                          
  val summand4 = computeSummand(x4, y4, theta)
  println("Summand 4:")
  println(summand4)
  val labeledPoint4 = createLabeledPoint(x4, y4)
  println("LabeledPoint 4:")
  println(labeledPoint4) 
}
main(Array())


// COMMAND ----------

//Part 2
import org.apache.spark.ml.linalg.{Vector, DenseVector}
import org.apache.spark.ml.feature.LabeledPoint
def predictLabel(w: Vector, observation: LabeledPoint): (Double, Double) = {
  val features = observation.features
  val actualLabel = observation.label
  val prediction = w dot features
  (actualLabel, prediction)
}
def main(args: Array[String]): Unit = {
  val w = new DenseVector(Array(0.5, 0.3, -0.2))
  val data = Seq(
    LabeledPoint(1.0, new DenseVector(Array(1.0, 2.0, 3.0))),
    LabeledPoint(2.0, new DenseVector(Array(4.0, 5.0, 6.0))),
    LabeledPoint(3.0, new DenseVector(Array(7.0, 8.0, 9.0)))
  )
  val predictions = data.map(observation => predictLabel(w, observation))
  predictions.foreach { case (actual, prediction) =>
    println(s"Actual Label: $actual, Predicted Label: $prediction")
  }
}
main(Array())


// COMMAND ----------

//Part 3
import org.apache.spark.ml.linalg.{Vector, DenseVector}
import org.apache.spark.ml.feature.LabeledPoint
def predictLabel(w: Vector, observation: LabeledPoint): (Double, Double) = {
  val features = observation.features
  val actualLabel = observation.label
  val prediction = w dot features
  (actualLabel, prediction)
}
def computeRMSE(predictions: Seq[(Double, Double)]): Double = {
  val squaredErrors = predictions.map { case (actual, prediction) =>
    math.pow(actual - prediction, 2)
  }
  math.sqrt(squaredErrors.sum / predictions.size)
}

def main(args: Array[String]): Unit = {
  val w = new DenseVector(Array(0.5, 0.3, -0.2))
  val data = Seq(
    LabeledPoint(1.0, new DenseVector(Array(1.0, 2.0, 3.0))),
    LabeledPoint(2.0, new DenseVector(Array(4.0, 5.0, 6.0))),
    LabeledPoint(3.0, new DenseVector(Array(7.0, 8.0, 9.0)))
  )
  val predictions = data.map(observation => predictLabel(w, observation))
  val rmse = computeRMSE(predictions)
  println(s"Root Mean Squared Error (RMSE): $rmse")
  val exampleRDD = sc.parallelize(Seq(
    (4.0, 4.5),
    (5.0, 5.2),
    (6.0, 6.8)
  ))
  val exampleRMSE = computeRMSE(exampleRDD.collect())
  println(s"Example Root Mean Squared Error (RMSE): $exampleRMSE")
}
main(Array())


// COMMAND ----------

//PART - 4:
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector => BDenseVector}

def toBreeze(v: Vector): BDenseVector[Double] = {
  BDenseVector(v.toArray)
}

def toSpark(v: BDenseVector[Double]): Vector = {
  Vectors.dense(v.toArray)
}

def elementWiseMultiply(v1: Vector, v2: Vector): Vector = {
  val multipliedArray = v1.toArray.zip(v2.toArray).map { case (x, y) => x * y }
  Vectors.dense(multipliedArray)
}

def computeSummand(x: Vector, y: Double, theta: Vector): Vector = {
  val xDense = toBreeze(x)
  val thetaDense = toBreeze(theta)
  val prediction = xDense dot thetaDense
  val error = prediction - y
  val summand = elementWiseMultiply(x, Vectors.dense(Array.fill(x.size)(error)))
  summand
}

def createLabeledPoint(features: Vector, label: Double): LabeledPoint = {
  LabeledPoint(label, features)
}

def gradientDescent(trainData: RDD[LabeledPoint], numIterations: Int): (Vector, Array[Double]) = {
  val n = numIterations.toDouble
  var alpha = 1.0
  var w = Vectors.zeros(trainData.first().features.size)

  val trainingErrors = new Array[Double](numIterations)

  for (i <- 0 until numIterations) {
    alpha = alpha / (n * math.sqrt(i+1))

    val gradients = trainData.map { labeledPoint =>
      val features = labeledPoint.features
      val prediction = w dot features
      val error = prediction - labeledPoint.label
      computeSummand(features, error, w)
    }.reduce((a, b) => Vectors.dense((toBreeze(a) + toBreeze(b)).toArray))

    w = Vectors.dense((toBreeze(w) - (toBreeze(gradients) * alpha)).toArray)

    val predictions = trainData.map(observation => (observation.label, w dot observation.features))
    val rmse = computeRMSE(predictions)
    trainingErrors(i) = rmse
  }

  (w, trainingErrors)
}

def computeRMSE(predictions: RDD[(Double, Double)]): Double = {
  val sumSquaredDifferences = predictions.map { case (actual, prediction) =>
    val difference = actual - prediction
    difference * difference
  }.sum()

  val meanSquaredError = sumSquaredDifferences / predictions.count()
  val rootMeanSquaredError = math.sqrt(meanSquaredError)
  rootMeanSquaredError
}

val data = Seq(
  LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 3.0)),
  LabeledPoint(2.0, Vectors.dense(4.0, 5.0, 6.0)),
  LabeledPoint(3.0, Vectors.dense(7.0, 8.0, 9.0))
)

val rdd = spark.sparkContext.parallelize(data)

val numIterations = 5

val (weights, trainingErrors) = gradientDescent(rdd, numIterations)

println("Final Weight Vector:")
println(weights)

println("Training Errors (RMSE) at each iteration:")
trainingErrors.reverse.foreach(println)

// COMMAND ----------

//Bonus

import org.apache.spark.ml.linalg.{Vector, DenseVector, DenseMatrix}
import org.apache.spark.ml.feature.LabeledPoint
import breeze.linalg.{DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector, pinv}

def closedFormSolution(X: DenseMatrix, y: DenseVector): DenseVector = {
  val Xbreeze = new BreezeDenseMatrix(X.numRows, X.numCols, X.toArray)
  val yBreeze = new BreezeDenseVector(y.toArray)
  val XtX = Xbreeze.t * Xbreeze
  val XtXInverse = pinv(XtX)
  val XtY = Xbreeze.t * yBreeze
  val weightsBreeze = XtXInverse * XtY
  new DenseVector(weightsBreeze.data)
}

def main(args: Array[String]): Unit = {
  // Sample LabeledPoint RDD with features and labels
  val data = Seq(
    LabeledPoint(1.0, new DenseVector(Array(1.0, 2.0, 3.0))),
    LabeledPoint(2.0, new DenseVector(Array(4.0, 5.0, 6.0))),
    LabeledPoint(3.0, new DenseVector(Array(7.0, 8.0, 9.0)))
  )

  // Convert data to a DenseMatrix and DenseVector
  val X = new DenseMatrix(data.length, data.head.features.size, data.flatMap(_.features.toArray).toArray)
  val y = new DenseVector(data.map(_.label).toArray)


  // Calculate weights using closed form solution
  val weights = closedFormSolution(X, y)

  // Print the final weights
  println("Final Weights:")
  println(weights)
}

// Call the main function
main(Array())

