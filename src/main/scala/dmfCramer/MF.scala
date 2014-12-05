package dmfCramer

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

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
  val U = DenseMatrix.rand(dimension, rows, new Uniform(-sigma, sigma))  // DenseMatrix.rand(size * dimension, rows, new Uniform(-sigma, sigma)) 
  val V = DenseMatrix.rand(size * dimension, cols, new Uniform(-sigma, sigma))
  
  // bias factor
  //val u0 = new DenseMatrix[Double](size, rows) 
  //val v0 = new DenseMatrix[Double](size, cols) 
  
  
  val theta: CSCMatrix[Double] = M.mapActiveValues(x => 0.0)
    
  /** compute the gradient w.r.t. the i-th column of U and j-th column of V */
  private def prob(i: Int, j: Int): DenseVector[Double] = {
    assert(i>=0 && i<rows && j>=0 && j<cols)

    val u = U(::, i)
    val v = V(::, j).asDenseMatrix.reshape(dimension, size)
    val s = sigmoid(sum((v(::,*) :* u).apply(::, *)).toDenseVector )  // :*= (exp(u0(::, i) + v0(::, j)))
    val p = s // exp(s)    
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
  
  private def project() = {

    U(*, ::) -= sum(U(::, *)).toDenseVector
    V(*, ::) -= sum(V(::, *)).toDenseVector
  }
  private def gradient(i: Int, j: Int, score: Int) : (DenseVector[Double], DenseVector[Double], DenseVector[Double], Double) = {
    val u = U(::, i) // U(::, i).asDenseMatrix.reshape(dimension, size)
    val v = V(::, j).asDenseMatrix.reshape(dimension, size)
    
    val uv = (v(::, *) :* u)
    val s = sigmoid(sum(uv(::, *)).toDenseVector)// += u0(::, i) += v0(::, j))
    val p = s.copy //exp(s);
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
    val ey = exp(y * ratio)
    val q = ey :* p
       
    val r = sum(q); 
    q /= r
    
    val dp = log(q) - log(p) - 1.0 - y * ratio
    //dp *= ( y * (f/df) += 1.0) // f should be small
    
    dp *=  (-p + 1.0) :*= (- s + 1.0) :*= p 
    
    
    val dU = V(::, j).asDenseMatrix.reshape(dimension, size, false)
    val dV = new DenseMatrix[Double](dimension, size)
    dV(::, *) := u

    dU(*, ::) :*= dp
    dV(*, ::) :*= dp
    (sum(dU(*, ::)).toDenseVector, dV.flatten(), dp, -log (r))
  }
  
   def solve() : Unit = {
    val dU = new DenseMatrix[Double](dimension, rows)
    val dV = new DenseMatrix[Double](size * dimension, cols)
//    val du0 = new DenseMatrix[Double](size, rows)
//    val dv0 = new DenseMatrix[Double](size, cols)
    
    var totalr: Double = 0
    var regr: Double = 0
    val delta = 0.01
    val activeSize = M.activeSize
    val regCoeff = 1/(sigma * sigma)
    for (iter <- 0 until 100) {
      // projection
      // project() 
      
      totalr = 0.0
      regr = regCoeff * (sum(U :* U) + sum(V :* V) ) / 2
      dU := 0d
      dV := 0d
//      du0 :=0d
//      dv0 :=0d
      M.activeIterator.foreach
	  {
	      case ((i,j), s) => {
	        val (dui, dvj, dpij, r) = gradient(i, j, s.toInt)
	        dU(::, i) += dui
	        dV(::, j) += dvj
//	        du0(::, i) += dpij
//	        dv0(::, j) += dpij
	        totalr += r
	  }
    }
      println(sqrt(sum(U:*U)), sqrt(sum(dU :* dU)) * delta)
      U -= (dU *= (delta)) 
      V -= (dV *= (delta))
      U += (U * (regCoeff * delta))
      V += (V * (regCoeff * delta))
    
    
//    u0 -= ((du0 *= (delta)) )
//    v0 -= ((dv0 *= (delta)) )
    println(iter, regr/activeSize, totalr/activeSize, (regr + totalr)/activeSize)
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