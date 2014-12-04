package dmfCramer

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Gaussian

/** general trait for the class of matrix factorization */
trait MF {

  val dimension: Int
  val rows, cols: Int
  val M: CSCMatrix[Double] // sparse rating matrix 
  
  def solve(): Unit
  def predict(i: Int, j: Int): Double
}


class discreteMF (val dimension: Int, val size: Int, val M:CSCMatrix[Double]) extends MF {
  val sigma = 1.0
  val rows = M.rows
  val cols = M.cols
  // factor
  val U = DenseMatrix.rand(size * dimension, rows, new Gaussian(0, sigma)) 
  val V = DenseMatrix.rand(size * dimension, cols, new Gaussian(0, sigma))
  
  // bias factor
  val u0 = DenseMatrix.rand(size, rows, new Gaussian(0, sigma)) 
  val v0 = DenseMatrix.rand(size, cols, new Gaussian(0, sigma)) 
  
  
  val theta: CSCMatrix[Double] = M.mapActiveValues(x => 0.0)
    
  /** compute the gradient w.r.t. the i-th column of U and j-th column of V */
  private def prob(i: Int, j: Int): DenseVector[Double] = {
    assert(i>=0 && i<rows && j>=0 && j<cols)

    val u = U(::, i).asDenseMatrix.reshape(dimension, size)
    val v = V(::, j).asDenseMatrix.reshape(dimension, size)
    val s = (sum((u :* v).apply(::, *)).toDenseVector += u0(::, i) += v0(::, j))
    val p = exp(s)    
    p /= (sum(p))
  }
  
  val eposilon = 1E-3
  val Y = {
    val t = (0 until size).map(i => DenseVector[Double]( (-i until (size - i)).map(_.toDouble).toArray))
    t.map(v => {t(0) += -eposilon})
    t.map(v => {t(size-1) += eposilon})
    t
  }
  
  val Y2 = Y.map(v => v :* v)
  // ** assuming score starts from 1*/
  private def gradient(i: Int, j: Int, score: Int) : (DenseVector[Double], DenseVector[Double], DenseVector[Double], Double) = {
    val u = U(::, i).asDenseMatrix.reshape(dimension, size)
    val v = V(::, j).asDenseMatrix.reshape(dimension, size)
    val s = (sum((u :* v).apply(::, *)).toDenseVector += u0(::, i) += v0(::, j))
    val p = exp(s);
    p /= (sum(p))

    val y = Y(score)
    val y2= Y2(score)
    // three Newton Raphson step
    var ratio = theta(i,j) 
    var f = sum(exp(y * ratio) :*= y :*= p)
    var df = sum(exp(y * ratio) :*= y2 :*= p)
    ratio = ratio - f/df
    for (j<- 0 until 10) {
    f = sum(exp(y * ratio) :*= y :*= p)
    df = sum(exp(y * ratio) :*= y2 :*= p)
    ratio = ratio - f/df
    }
    //assert(!ratio.isNaN())
    theta(i,j) = ratio
    
    // compute gradient
    val dp = exp(y * ratio)
    val r = sum(dp :* p); 
    
    dp /= (-r) 
    dp *= ( y * (f/df) += 1.0) *= (p - p :* p) // f should be small
    val dU = V(::, j).asDenseMatrix.reshape(dimension, size, false)
    val dV = U(::, i).asDenseMatrix.reshape(dimension, size, false)
    dU(*, ::) :*= dp
    dV(*, ::) :*= dp
    (dU.flatten(), dV.flatten(), dp, -log (r))
  }
  
   def solve() : Unit = {
    val dU = new DenseMatrix[Double](size * dimension, rows)
    val dV = new DenseMatrix[Double](size * dimension, cols)
    val du0 = new DenseMatrix[Double](size, rows)
    val dv0 = new DenseMatrix[Double](size, cols)
    var totalr: Double = 0
    val delta = 0.1
    val activeSize = M.activeSize
    val regCoeff = 0.1
    for (iter <- 0 until 100) {
      totalr = regCoeff * (sum(U :* U) + sum(V :* V) + sum(u0 :* u0) + sum(v0 :* v0)) / 2
      dU := 0d
      dV := 0d
      du0 :=0d
      dv0 :=0d
      M.activeIterator.foreach
	  {
	      case ((i,j), s) => {
	        val (dui, dvj, dpij, r) = gradient(i, j, s.toInt)
	        dU(::, i) += dui
	        dV(::, j) += dvj
	        du0(::, i) += dpij
	        dv0(::, j) += dpij
	        totalr += r
	  }
    }
    U -= (dU * (delta/activeSize) + U * (regCoeff * delta))
    V -= (dV * (delta/activeSize) + V * (regCoeff * delta))
    u0 -= (du0 * (delta/activeSize) + u0 * (regCoeff * delta))
    v0 -= (dv0 * (delta/activeSize) + v0 * (regCoeff * delta))
    println(iter, totalr/activeSize)
    }
  }

  
  val scoreVector = DenseVector[Double]((1 to size).map(_.toDouble).toArray)
  def predict(i: Int, j: Int) = (prob(i,j) dot scoreVector)
  
}


object discreteMFrun {
  def main(argv: Array[String]): Unit = {
    val builder = new CSCMatrix.Builder[Double](rows = 7000, cols = 4000)
    import scala.io.Source._
    val source = scala.io.Source.fromFile("train_vec.txt")
    val lines = source.getLines()
    lines.foreach(s => {
      val elem = s.split("\\s+")
      if (elem.size >= 4) builder.add(elem(1).toInt, elem(2).toInt, elem(3).toDouble)
    })
    source.close()
    val M = builder.result()
    println(M.activeSize)
    val dmf = new discreteMF(10, 7, M)
    dmf.solve()
    
  }
}