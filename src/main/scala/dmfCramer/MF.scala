/**
  Discrete Martrix Factorizatin With Cramer Risk Minimization
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

  
  val dirichlet = DenseVector.zeros[Double](size)
  val M: CSCMatrix[Double] = {
    val r = L.maxBy(_._1)._1+1
    val c = L.maxBy(_._2)._2+1
    println(r, c, L.length)
    val builder = new CSCMatrix.Builder[Double](rows = r, cols = c)
    L.foreach{case (i,j, s)=>{
      builder.add(i, j, s)
      dirichlet(s.toInt) = dirichlet(s.toInt) + 1
    }}
    (dirichlet -= 1.0) /= (L.length.toDouble / size)
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
      abs(v - x._3)}).sum / tB.length / 1.6 // 1.6 is the E[MAE] for 1..5

    // min/max measurement
    /*
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
     */

    println(RMSE, NMAE)
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
      I(DenseVector.rand(dimension, new Bernoulli(0.9)))
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
    
    val diri = dirichlet * 0.001
    val dp = (p - q) += ((p - 1.0) *= diri)
    
    
    val dU = V(::, j).asDenseMatrix.reshape(dimension, size, false)
    val dV = new DenseMatrix[Double](dimension, size)
    dV(::, *) := u

    dU(*, ::) :*= dp
    dV(*, ::) :*= dp

    if (useDropout)
      (sum(dU(*, ::)).toDenseVector :* dropout, dV.flatten(), dp, -log (r) - sum(log(p) :*= diri))
    else
      (sum(dU(*, ::)).toDenseVector, dV.flatten(), dp, -log (r) - sum(log(p) :*= diri))
  }
  
  def solve(delta0: Double = 0.1, // initial learning rate
            momentum: Double = 0.9, //
            batchSize: Int = 10000,
            regCoeff: Double = 0.00,
            numOfEpoches: Int = 50,
            useDropout: Boolean = false) : Unit = {
    val dU = new DenseMatrix[Double](dimension, rows)
    val dV = new DenseMatrix[Double](size * dimension, cols)
    //val du0 = new DenseMatrix[Double](size, rows)
    val dv0 = new DenseMatrix[Double](size, cols)
    


    val activeSize = L.length

    println("epoch\treg\trisk\tloss")

    for (iter <- 0 until numOfEpoches) {
      val delta = delta0 * scala.math.pow(0.01, iter / numOfEpoches.toDouble)
      var totalr = 0.0
      var regr = 0.0 // regCoeff * (sum(U :* U) + sum(V :* V) ) / 2f
      val batches = util.Random.shuffle(L).grouped(batchSize).toList // shuffling samples and grouped into batches
      batches.foreach(batch => {
        dU *= momentum
        dV *= momentum
        //du0 :=0d
        dv0 *=momentum
        totalr += batch.par.map{
          case (i, j, s) => {
	    val (dui, dvj, dpij, r) = gradient(i, j, s.toInt, useDropout)
	    dU(::, i) += dui
	    dV(::, j) += dvj
	    //du0(::, i) += dpij
	    dv0(::, j) += dpij
	    //println(i, j, s)
	    r
	    }}.reduce(_+_)
        //println(sqrt(sum(U:*U)), sqrt(sum(dU :* dU)) * delta)
        U -= (dU *= (delta))
        V -= (dV *= (delta))
        U -= (U * (regCoeff * delta))
        V -= (V * (regCoeff * delta))
        //u0 -= ((du0 *= (delta)) )
        v0 -= (dv0 *= (delta))
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
  
  
  def save(filename: String): Unit = {
    import com.jmatio.io._
    import com.jmatio.types._
    import java.util.ArrayList
    
    val list = new ArrayList[MLArray]()
    list.add(new MLDouble("U", U.t.flatten(false).data, U.cols))
    list.add(new MLDouble("V", V.t.flatten(false).data, V.cols))
    list.add(new MLDouble("v0", v0.t.flatten(false).data, v0.cols))
    new MatFileWriter(filename, list)       
  }
  
  def load(filename: String): Unit = {
    import com.jmatio.io.MatFileReader
    import com.jmatio.types._
    import java.util.ArrayList    
    implicit def wrapDoubleArray(arr: Array[Array[Double]]): Array[Double] = arr.flatMap(x => x)
    val mfr = new MatFileReader( filename )
    val Us: DenseMatrix[Double] =  new DenseMatrix(U.rows, U.cols, mfr.getMLArray("U").asInstanceOf[MLDouble].getArray())
    val Vs: DenseMatrix[Double] =  new DenseMatrix(V.rows, V.cols, mfr.getMLArray("V").asInstanceOf[MLDouble].getArray())
    val v0s: DenseMatrix[Double] =  new DenseMatrix(v0.rows,v0.cols, mfr.getMLArray("v0").asInstanceOf[MLDouble].getArray())
    U := Us
    V := Vs
    v0 := v0s
  }

  private def probQ(i: Int, j: Int, theta: Double, cutoff: Int): (Int, Double) = {
    val u = U(::, i)
    val v = V(::, j).asDenseMatrix.reshape(dimension, size)
    val p = exp(sum((v(::,*) :* u).apply(::, *)).toDenseVector += v0(::, j))
    p /= (sum(p))
    val p1 = p(cutoff until size).sum
    // val theta = log(((1 - p1) * (1-value)) / (p1 * value))
    val q1 = p1 / (p1 + (1 - p1) * exp(theta) )
    val sample = I(new Bernoulli(q1).draw())
    (sample.toInt, log (p1 + (1 - p1)* exp(- theta)))
  }
  // sort itemList based on its top k probability
  def topk(user: Int, k: Int, itemList: IndexedSeq[Int] = (1 until V.cols)): IndexedSeq[Int] = {
    assert(user < rows)
    val scores = IndexedSeq.fill[Double](cols)(0)
    val N = itemList.length
    
    // Importance sampling to estimate L_{m} and L_{m-1}
    val sampleSize = 1000
    val pA = itemList.map(prob(user, _)(size - 1)).sum / N
    val pB = pA + itemList.map(prob(user, _)(size - 2)).sum / N
    val value: Double = k.toDouble / N.toDouble
    val theta = 0.77 * log ((1 - pA) * (1 - value) / ( pA * value)) // rescale
    val eA = for (i<- 0 until sampleSize) yield {
      val trials = itemList.map(probQ(user, _, theta, size-1)); 
      trials.foldLeft[Int](0)(_ + _._1) 
    }
    
    val eB = for (i<- 0 until sampleSize) yield {
      val trials = itemList.map(probQ(user, _, theta, size-2)); 
      trials.foldLeft[Int](0)(_ + _._1)
    }    
    println((pA, theta), (pB, theta))
    println(eA.filter(_ < k).length, eB.filter(_ < k).length)
    
    itemList.sortBy(scores(_))
  }
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
    dmf.save("DPMF.mat")    
    
    /*
    dmf.load("DPMF.mat")
    dmf.test(tB)
    
    dmf.topk(30, 10)
    */
  }
}
