/**
  Discrete Martrix Factorizatin With Cramer Risk
  Jianbo Ye (c) 2014-2015 
  
  Input: 
  - A sparse matrix with values taking discrete values from 1 .. M, each row is a user, and each column is an item. 
  - A representation dimension for user and item factors
  
  Output: 
  - A probability matrix for each cell (i,j) with a nonzero probability for discrete value from 0 ... (M+1), where 0 and (M+1) are considering as the "extrame values". One can either use the expected value as predictions or rank each row using extrame values.
  
  Reference
  Jianbo Ye, Top-k Probability Estimation Using Discrete Martrix Factorizatin: A Cram\"er Risk Minimization Approach (to appear)
 */

package dmfCramer

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

/** general trait for the class of matrix factorization */
trait MF {

  val dimension: Int
  val rows, cols: Int
  val M: CSCMatrix[Double] // sparse rating matrix
 
  def predict(i: Int, j: Int): Double   

}


class discreteMF (val dimension: Int, val size: Int,
  val L: List[(Int, Int, Double)], val tL: List[(Int, Int, Double)]) extends MF {
  val sigma = 1.0

  val M: CSCMatrix[Double] = {
    val r = L.maxBy(_._1)._1+1
    val c = L.maxBy(_._2)._2+1
    println(r, c, L.length)
    val builder = new CSCMatrix.Builder[Double](rows = r, cols = c)
    L.foreach{case (i,j, s)=>builder.add(i, j, s)}
    builder.result()
  }
  
  val rows = M.rows
  val cols = M.cols
  val theta = M.mapActiveValues(x=> 0.0)
  
  // factor
  val U = DenseMatrix.rand(dimension, rows, new Uniform(-sigma, sigma))  // DenseMatrix.rand(size * dimension, rows, new Uniform(-sigma, sigma))
  val V = DenseMatrix.rand(size * dimension, cols, new Uniform(-sigma, sigma))
  
  // bias factor
  //val u0 = new DenseMatrix[Double](size, rows)
  val v0 = new DenseMatrix[Double](size, cols)


  /** testing a new tM */
  def test(tB: List[(Int, Int, Double)]): Unit = {

    val maxV: Double = size - 2
    val minV: Double = 1

    // usual measurement
    val RMSE = sqrt(tB.map(x => {
        val v = min(max(predict(x._1,x._2), minV), maxV)
        val err = (v - x._3); err * err }).sum / tB.length)
    val NMAE = tB.map(x => {
        val v = min(max(predict(x._1,x._2), minV), maxV)
        abs(v - x._3)}).sum / tB.length

    // min/max measurement
    val maxSAE = {
      val ftB = tB.filter(_._3 == maxV).map(x => {
        val v = min(max(predict(x._1,x._2), minV), maxV)
        x._3 - v})
      ftB.sum / ftB.length
    }
    val minSAE = {
      val ftB = tB.filter(_._3 == 1).map(x => {
        val v = min(max(predict(x._1,x._2), minV), maxV)
        v - x._3})
      ftB.sum/ ftB.length
    }


    println(RMSE, NMAE, maxSAE, minSAE)
  }
  
  /** compute the gradient w.r.t. the i-th column of U and j-th column of V */
  private def prob(i: Int, j: Int): DenseVector[Double] = {
    assert(i>=0 && i<rows && j>=0 && j<cols)

    val u = U(::, i)
    val v = V(::, j).asDenseMatrix.reshape(dimension, size)
    val s = exp(sum((v(::,*) :* u).apply(::, *)).toDenseVector += v0(::, j))
    s /= (sum(s))
  }
  
  val Y = (0 until size).map(i => 
    DenseVector[Double]( (-i until (size - i)).map(_.toDouble).toArray))
  val Y2 = Y.map(v => v :* v)
  
  // ** assuming score starts from 1*/
  private def gradient(i: Int, j: Int, score: Int, useDropout: Boolean = false) : (DenseVector[Double], DenseVector[Double], DenseVector[Double], Double) = {
    val dropout = if (useDropout) {
      I(DenseVector.rand(dimension, new Bernoulli(0.5)))
    } else null

    val u = if (useDropout) {
      U(::, i) :* dropout
    } else U(::, i) // U(::, i).asDenseMatrix.reshape(dimension, size)

    val v = V(::, j).asDenseMatrix.reshape(dimension, size)
    
    val s = exp(sum((v(::, *) :* u).apply(::, *)).toDenseVector += v0(::, j))
    val p = s.copy //exp(s);
    p /= (sum(p))

    val y = Y(score)
    val yp = y :* p
    val y2p= Y2(score) :* p
    
    // Newton Raphson step for ratio
    var ratio = theta(i,j) // hot-start from cached theta
    var f = sum(exp(y * ratio) :*= yp)
    var df = sum(exp(y * ratio) :*= y2p)
    ratio = ratio - f/df
    for (j<- 0 until 5) {
      f = sum(exp(y * ratio) :*= yp)
      df = sum(exp(y * ratio) :*= y2p)
      ratio = ratio - f/df
    }
    //assert(!ratio.isNaN())
    theta(i,j) = ratio // caching
    
    // compute gradient
    val ey = exp(y * ratio)
    val q = ey :*= p
    
    val r = sum(q) // risk
    q /= r
    
    val dp = p -= q
    
    
    val dU = V(::, j).asDenseMatrix.reshape(dimension, size, false)
    val dV = new DenseMatrix[Double](dimension, size)
    dV(::, *) := u

    dU(*, ::) :*= dp
    dV(*, ::) :*= dp

    if (useDropout)
      (sum(dU(*, ::)).toDenseVector :* dropout, dV.flatten(), dp, -log (r))
    else
      (sum(dU(*, ::)).toDenseVector, dV.flatten(), dp, -log (r))
  }
  
  def solve(delta0: Double = 0.1, // initial learning rate
            momentum: Double = 0.9, //
            batchSize: Int = 10000,
            regCoeff: Double = 0.001,
            numOfEpoches: Int = 500,
            useDropout: Boolean = true) : Unit = {
    val dU = new DenseMatrix[Double](dimension, rows)
    val dV = new DenseMatrix[Double](size * dimension, cols)
    //val du0 = new DenseMatrix[Double](size, rows)
    val dv0 = new DenseMatrix[Double](size, cols)
    
    var totalr: Double = 0
    var regr: Double = 0
    var delta = delta0 // learning rate
    val activeSize = L.length

    println("epoch\treg\trisk\tloss")

    for (iter <- 0 until numOfEpoches) {
      delta = delta0 * scala.math.pow(0.01, iter / numOfEpoches.toDouble)
      totalr = 0.0
      regr = if (!useDropout) regCoeff * (sum(U :* U) + sum(V :* V) ) / 2 else 0.0
      val batches = util.Random.shuffle(L).grouped(batchSize).toList // shuffling samples and grouped into batches
      batches.foreach(batch => {
        dU *= momentum
        dV *= momentum
        //du0 :=0d
        dv0 *=momentum
        batch.foreach{
          case (i, j, s) => {
	    val (dui, dvj, dpij, r) = gradient(i, j, s.toInt, useDropout)
	    dU(::, i) += dui
	    dV(::, j) += dvj
	    //du0(::, i) += dpij
	    dv0(::, j) += dpij
	    //println(i, j, s)
	    totalr += r
	  }}
        //println(sqrt(sum(U:*U)), sqrt(sum(dU :* dU)) * delta)
        U -= (dU *= (delta))
        V -= (dV *= (delta))
        U -= (U * (regCoeff * delta))
        V -= (V * (regCoeff * delta))
        //u0 -= ((du0 *= (delta)) )
        //v0 -= (dv0 *= (delta))
      })
      println("%d\t%f\t%f\t%f".format( iter, 
        regr/batchSize, (totalr/activeSize),  (regr/batchSize + totalr/activeSize) ))
      if (iter % 5 == 0) {
        test(L)
        test(tL)
      }
      
    }
  }

  
  val scoreVector = DenseVector[Double]((0 until size).map(_.toDouble).toArray)
  def predict(i: Int, j: Int) = (prob(i,j) dot scoreVector)
  

  
}


object discreteMFrun {
  def getList(filename: String): List[(Int, Int, Double)] = {
    //val builder = new CSCMatrix.Builder[Double](rows = 7000, cols = 4000)
    import scala.io.Source._
    val source = scala.io.Source.fromFile(filename)
    val lines = source.getLines()
    import scala.collection.mutable.ListBuffer
    val lb: ListBuffer[(Int, Int, Double)] = ListBuffer[(Int, Int, Double)]()
    lines.foreach{s =>
      val elem = s.split("\\s+")
      if (elem.size>=4)
        lb.+=((elem(1).toInt, elem(2).toInt, elem(3).toDouble))
    }
    source.close()
    lb.toList
    //builder.result()
  }
  
  def main(argv: Array[String]): Unit = {

    val B = getList("train_vec.txt")
    val tB= getList("probe_vec.txt")
    val dmf = new discreteMF(10, 7, B, tB)
    
    dmf.solve()    
  }
}
