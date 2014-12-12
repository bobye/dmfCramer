/**
  Discrete Martrix Factorizatin With Cramer Risk Minimization
  Jianbo Ye (c) 2014-2015 
  
  Input: 
  - A sparse matrix with values taking discrete values from 1 .. M, each row is a user, and each column is an item. 
  - A representation dimension for user and item factors
  
  Output: 
  - A probability matrix for each cell (i,j) with a nonzero probability for discrete value from 0 ... (M+1), where 0 and (M+1) are considering as the "extrame values". One can either use the expected value as predictions or rank each row using extrame values.
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
    (dirichlet -= 1.0) /= (L.length.toDouble / size) *= 0.0001
    builder.result()    
  }

  val rows = M.rows
  val cols = M.cols
  val theta = M.mapActiveValues(x=> 0.0)
  
  // factor
  val U = DenseMatrix.rand(dimension, rows, new Uniform(-sigma, sigma))  // DenseMatrix.rand(size * dimension, rows, new Uniform(-sigma, sigma))
  val V = DenseMatrix.rand(size * dimension, cols, new Uniform(-sigma, sigma))
  
  // bias factor
  // val u0 = new DenseMatrix[Double](size, rows)
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
  val Y2c = Y2.map( v => {v - v.sum / size})
  
  // ** assuming score starts from 1*/
  private def gradient(i: Int, j: Int, score: Int, useDropout: Boolean = false) : 
	  (DenseVector[Double], DenseVector[Double], DenseVector[Double], (Double,Double,Double)) = {
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
    
    val coeff = 1.0
    val dp = (p - q) // += ((p - 1.0) *= dirichlet) -= ((p - 1.0) :*= Y2c(score) :* coeff :*= p)
    
    
    val dU = V(::, j).asDenseMatrix.reshape(dimension, size, false)
    val dV = new DenseMatrix[Double](dimension, size)
    dV(::, *) := u

    dU(*, ::) :*= dp
    dV(*, ::) :*= dp

    
    val obj1 = -log (r) 
    val obj2 = 0.0 //- sum(log(p) :*= dirichlet) 
    val obj3 = 0.0 // sum(p :* Y2(score)) * coeff
    if (useDropout)
      (sum(dU(*, ::)).toDenseVector :* dropout, dV.flatten(), dp, (obj1, obj2, obj3))
    else
      (sum(dU(*, ::)).toDenseVector, dV.flatten(), dp, (obj1, obj2, obj3))
  }
  
  def solve(delta0: Double = 0.1, // initial learning rate
            momentum: Double = 0.9, //
            batchSize: Int = 10000,
            regCoeff: Double = 0.02,
            numOfEpoches: Int = 500,
            useDropout: Boolean = false) : Unit = {
    val dU = new DenseMatrix[Double](dimension, rows)
    val dV = new DenseMatrix[Double](size * dimension, cols)
    //val du0 = new DenseMatrix[Double](size, rows)
    val dv0 = new DenseMatrix[Double](size, cols)
    


    val activeSize = L.length

    println("epoch\treg1\treg2\trisk\tloss")

    for (iter <- 0 until numOfEpoches) {
      val delta = delta0 * scala.math.pow(0.01, iter / numOfEpoches.toDouble)
      var totalr1 = 0.0
      var totalr2 = 0.0
      var regr1 = regCoeff * (sum(U :* U) + sum(V :* V) ) / 2f
      var regr2 = 0.0
      val batches = util.Random.shuffle(L).grouped(batchSize).toList // shuffling samples and grouped into batches
      batches.foreach(batch => {
        dU *= momentum
        dV *= momentum
        //du0 :=0d
        dv0 *=momentum
        val totalR = batch.par.map{
          case (i, j, s) => {
	    val (dui, dvj, dpij, r) = gradient(i, j, s.toInt, useDropout)
	    dU(::, i) += dui
	    dV(::, j) += dvj
	    //du0(::, i) += dpij
	    dv0(::, j) += dpij
	    //println(i, j, s)
	    r
	    }}.reduce((a, b) => (a._1+b._1, a._2+b._2, a._3+b._3))
	    totalr1 += totalR._1
	    totalr2 += totalR._3
	    regr2 += totalR._2
        //println(sqrt(sum(U:*U)), sqrt(sum(dU :* dU)) * delta)
        U -= (dU *= (delta))
        V -= (dV *= (delta))
        U -= (U * (regCoeff * delta))
        V -= (V * (regCoeff * delta))
        //u0 -= ((du0 *= (delta)) )
        v0 -= (dv0 *= (delta))
      })
      println("%d\t%f\t%f\t%f\t%f".format( iter, regr1 / batchSize,
        regr2/activeSize, (totalr1/activeSize),  (totalr2/activeSize) ))
      if (iter % 5 == 0) {
        test(L)
        test(tL)
      }
      
    }
  }

  
  val scoreVector = DenseVector[Double]((0 until size).map(_.toDouble).toArray)
  def predict(i: Int, j: Int) = {
    (prob(i,j) dot scoreVector)
  }
  
  
  def save(filename: String): Unit = {
    import com.jmatio.io._
    import com.jmatio.types._
    import java.util.ArrayList
    
    val list = new ArrayList[MLArray]()
    list.add(new MLDouble("U", U.t.flatten(false).data, U.cols))
    list.add(new MLDouble("V", V.t.flatten(false).data, V.cols))
    //list.add(new MLDouble("u0", u0.t.flatten(false).data, u0.cols))
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
    //val u0s: DenseMatrix[Double] =  new DenseMatrix(u0.rows,u0.cols, mfr.getMLArray("u0").asInstanceOf[MLDouble].getArray())
    val v0s: DenseMatrix[Double] =  new DenseMatrix(v0.rows,v0.cols, mfr.getMLArray("v0").asInstanceOf[MLDouble].getArray())
    U := Us
    V := Vs
    //u0 := u0s
    v0 := v0s
  }
  
  def exportResult(filename: String, userList: IndexedSeq[Int], itemList: IndexedSeq[Int]): Unit = {
    val Ptotal = DenseMatrix.zeros[Double](rows, cols)
    val Pmax = DenseMatrix.zeros[Double](rows, cols)
    for (i<- 0 until size) {
        val Q = (U.t * V(i*dimension until (i+1)*dimension, ::))
        //Q(::, *) += u0(i, ::).t
        Q(*, ::) += v0(i, ::).t
        val Qexp = exp(Q)
        if (i == (size - 1))
          Pmax := Qexp
    	Ptotal += Qexp
    }
    Pmax /= Ptotal
    val Pselected: DenseMatrix[Double] = Pmax(userList, itemList).toDenseMatrix
    import com.jmatio.io._
    import com.jmatio.types._
    import java.util.ArrayList
    val list = new ArrayList[MLArray]()
    list.add(new MLDouble("pmax", Pselected.data, Pselected.rows))
    println("export pmax to " + filename + " ... ")
    new MatFileWriter(filename, list) 
    println("[done]")
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
    (sample.toInt, log (p1 + (1 - p1)* exp(theta)))
  }
  // sort itemList based on its top k probability
  def topk(user: Int, k: Int, itemList: IndexedSeq[Int] = (1 until V.cols)): IndexedSeq[Int] = {
    assert(user < rows)
    val scores = IndexedSeq.fill[Double](cols)(0)
    val N = itemList.length
    
    // Importance sampling to estimate L_{m} and L_{m-1}
    val sampleSize = 1000
    
    import com.jmatio.io._
    import com.jmatio.types._
    import java.util.ArrayList
    
    val list = new ArrayList[MLArray]()
    
    val pscore = itemList.map(prob(user, _)(size - 1)).toArray
    val escore = itemList.map(predict(user, _)).toArray
    list.add(new MLDouble("pscore", pscore, pscore.length))
    list.add(new MLDouble("escore", escore, escore.length))
    new MatFileWriter("userana.mat", list) 

    val pA = itemList.map(prob(user, _)(size - 1)).sum / N
    val pB = pA + itemList.map(prob(user, _)(size - 2)).sum / N
    val value: Double = k.toDouble / N.toDouble
    val thetaA = 0.77 * log ((1 - pA) * (1 - value) / ( pA * value)) // rescale
    val thetaB = thetaA // 0.9 * log ((1 - pB) * (1 - value) / ( pB * value)) // rescale
    
    val eA = for (i<- 0 until sampleSize) yield {
      val trials = itemList.map(probQ(user, _, thetaA, size-1)); 
      (trials.foldLeft[Int](0)(_ + _._1), trials.foldLeft[Double](0)(_ + _._2))
    }
    
    val eB = for (i<- 0 until sampleSize) yield {
      val trials = itemList.map(probQ(user, _, thetaB, size-2)); 
      (trials.foldLeft[Int](0)(_ + _._1), trials.foldLeft[Double](0)(_ + _._2))
    }    
    println((pA, thetaA), (pB, thetaB))
    
    val countA = eA.filter(_._1 <= k).length
    val countB = eB.filter(_._1 <= k).length 
    println(countA, countB, - eA(0)._2 + eB(0)._2)
    
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
    
    dmf.load("DPMF.mat")
    dmf.test(tB)
    dmf.exportResult("DPMFresult.mat", 1 until dmf.rows, 1 until dmf.cols)
    
    dmf.topk(10, 10)
    
  }
}
