#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace py = pybind11;

// Example autodiff cost function from ceres tutorial
struct ExampleFunctor {
  template<typename T>
  bool operator()(const T *const x, T *residual) const {
    residual[0] = T(10.0) - x[0];
    return true;
  }

  static ceres::CostFunction *Create() {
    return new ceres::AutoDiffCostFunction<ExampleFunctor,
                                           1,
                                           1>(new ExampleFunctor);
  }
};

// cSE3 conversion functions: to/from pointer/vector
template<typename T>
void cSE3convert(const T* data, Sophus::SE3<T>& X) {
  X = Sophus::SE3<T>(Eigen::Quaternion<T>(data+3),
                     Eigen::Map<const Eigen::Matrix<T,3,1>>(data));
}
template<typename T>
void cSE3convert(const Eigen::Matrix<T,7,1> &Xvec, Sophus::SE3<T>& X) {
  X = Sophus::SE3<T>(Eigen::Quaternion<T>(Xvec.data()+3),
                     Eigen::Map<const Eigen::Matrix<T,3,1>>(Xvec.data()));
}
template<typename T>
void cSE3convert(const Sophus::SE3<T>& X, Eigen::Matrix<T,7,1> &Xvec) {
  Xvec.template block<3,1>(0,0) = Eigen::Matrix<T,3,1>(X.translation().data());
  Xvec.template block<4,1>(3,0) = Eigen::Matrix<T,4,1>(X.so3().data());
}

// AutoDiff cost function (factor) for the difference between a measured 3D
// relative transform, Xij = (tij_, qij_), and the relative transform between two  
// estimated poses, Xi_hat and Xj_hat. Weighted by measurement covariance, Qij_.
class cSE3Functor
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // store measured relative pose and inverted covariance matrix
  cSE3Functor(Eigen::Matrix<double,7,1> &Xij_vec, Eigen::Matrix<double,6,6> &Qij)
  {
    Sophus::SE3<double> Xij;
    cSE3convert(Xij_vec, Xij);
    Xij_inv_ = Xij.inverse();
    Qij_inv_ = Qij.inverse();
  }

  // templated residual definition for both doubles and jets
  // basically a weighted implementation of boxminus using Eigen templated types
  template<typename T>
  bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
  {
    // assign memory to usable objects
    Sophus::SE3<T> Xi_hat, Xj_hat;
    cSE3convert(_Xi_hat, Xi_hat);
    cSE3convert(_Xj_hat, Xj_hat);
    Eigen::Map<Eigen::Matrix<T,6,1>> r(_res);

    // compute current estimated relative pose
    const Sophus::SE3<T> Xij_hat = Xi_hat.inverse() * Xj_hat;

    // compute residual via boxminus (i.e., the logarithmic map of the error pose)
    // weight with inverse covariance
    r = Qij_inv_.cast<T>() * (Xij_inv_.cast<T>() * Xij_hat).log();        

    return true;
  }

  // cost function generator--ONLY FOR PYTHON WRAPPER
  static ceres::CostFunction *Create(Eigen::Matrix<double,7,1> &Xij, Eigen::Matrix<double,6,6> &Qij) {
    return new ceres::AutoDiffCostFunction<cSE3Functor,
                                           6,
                                           7,
                                           7>(new cSE3Functor(Xij, Qij));
  }

private:
  Sophus::SE3<double> Xij_inv_;
  Eigen::Matrix<double,6,6> Qij_inv_;
};

// AutoDiff cost function (factor) for the difference between a range measurement
// rij, and the relative range between two estimated poses, Xi_hat and Xj_hat. 
// Weighted by measurement variance, qij_.
class RangeFunctor
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // store measured range and inverted variance
  RangeFunctor(double &rij, double &qij)
  {
    rij_ = rij;
    qij_inv_ = 1.0 / qij;
  }

  // templated residual definition for both doubles and jets
  template<typename T>
  bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
  {
    // assign memory to usable objects
    Eigen::Matrix<T,3,1> ti_hat(_Xi_hat), tj_hat(_Xj_hat);

    // range measurement error, scaled by inverse variance
    *_res = static_cast<T>(qij_inv_) * (static_cast<T>(rij_) - (tj_hat - ti_hat).norm());

    return true;
  }

  // cost function generator--ONLY FOR PYTHON WRAPPER
  static ceres::CostFunction *Create(double &rij, double &qij) {
    return new ceres::AutoDiffCostFunction<RangeFunctor,
                                           1,
                                           7,
                                           7>(new RangeFunctor(rij, qij));
  }

private:
  double rij_;
  double qij_inv_;
};

// AutoDiff cost function (factor) for the difference between a measured 3D
// relative transform, Xij_ = (tij_, qij_), and the relative transform between two  
// estimated poses, Xi_hat and Xj_hat. Weighted by measurement covariance, Qij_.
class Transform3DFunctor
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // store measured relative pose and covariance matrix
  Transform3DFunctor(Eigen::Matrix<double,7,1> &X, Eigen::Matrix<double,6,6> &Q)
  {
    tij_ = X.block<3,1>(0,0);
    qij_ = Eigen::Quaterniond(X.block<4,1>(3,0));
    // residual weighted by inverse covariance
    Wij_ = Q.inverse();
  }

  // templated residual definition for both doubles and jets
  // basically a weighted implementation of boxminus using Eigen templated types
  template<typename T>
  bool operator()(const T* Xi_hat, const T* Xj_hat, T* res) const
  {
    // assign memory to usable objects
    Eigen::Map<const Eigen::Matrix<T,3,1>> ti_hat(Xi_hat);
    Eigen::Quaternion<T> qi_hat(Xi_hat+3);
    Eigen::Map<const Eigen::Matrix<T,3,1>> tj_hat(Xj_hat);
    Eigen::Quaternion<T> qj_hat(Xj_hat+3);
    Eigen::Map<Eigen::Matrix<T,6,1>> r(res);

    // compute trivial vector subtraction component
    Eigen::Matrix<T,3,1> t_res = tij_ - (tj_hat - ti_hat);

    // compute quaternion difference component
    Eigen::Quaternion<T> qij_hat = qi_hat.inverse() * qj_hat;
    Eigen::Quaternion<T> q_err = qij_hat.inverse() * static_cast<Eigen::Quaternion<T>>(qij_);
    Eigen::Matrix<T,3,1> q_res;
    if (q_err.w() < static_cast<T>(0.0))
    {
      q_err.coeffs() *= static_cast<T>(-1.0);
    }
    const Eigen::Matrix<T,3,1> qv(q_err.vec());
    T sinha = qv.norm();
    if (sinha > static_cast<T>(0.0))
    {
      T angle = static_cast<T>(2.0) * atan2(sinha, q_err.w());
      q_res = qv * (angle / sinha);
    }
    else
    {
      q_res = qv * (static_cast<T>(2.0) / q_err.w());
    }

    // assign to residual and weight by covariance
    r.template block<3,1>(0,0) = t_res;
    r.template block<3,1>(3,0) = q_res;
    r = Wij_ * r;

    return true;
  }

  // cost function generator--ONLY FOR PYTHON WRAPPER
  static ceres::CostFunction *Create(Eigen::Matrix<double,7,1> &X, Eigen::Matrix<double,6,6> &Q) {
    return new ceres::AutoDiffCostFunction<Transform3DFunctor,
                                           6,
                                           7,
                                           7>(new Transform3DFunctor(X, Q));
  }

private:
  Eigen::Matrix<double,3,1> tij_;
  Eigen::Quaternion<double> qij_;
  Eigen::Matrix<double,6,6> Wij_;
};

// AutoDiff cost function (factor) for the difference between a measured quaternion, 
// q_, and an estimated quaternion, q_hat.
class QuaterniondFunctor
{
public:
  // store internal measured quaternion
  QuaterniondFunctor(Eigen::Vector4d &x)
  {
    q_ = Eigen::Quaternion<double>(x.data());
    q_inv_ = q_.inverse();
  }

  // templated residual definition for both doubles and jets
  template<typename T>
  bool operator()(const T* _q_hat, T* res) const
  {
    Eigen::Quaternion<T> q_hat(_q_hat);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> r(res);

    // implementation of boxminus using templated Eigen types
    // https://eigen.tuxfamily.org/bz/show_bug.cgi?id=1244
    //  q_err = q^{-1} * q_hat
    Eigen::Quaternion<T> q_err(static_cast<Eigen::Quaternion<T>>(q_inv_)*(q_hat));
    //  q_err -> theta vector
    if (q_err.w() < static_cast<T>(0.0))
    {
      q_err.coeffs() *= static_cast<T>(-1.0);
    }
    const Eigen::Matrix<T,3,1> qv(q_err.vec());
    T sinha = qv.norm();
    if (sinha > static_cast<T>(0.0))
    {
      T angle = static_cast<T>(2.0) * atan2(sinha, q_err.w());
      r = qv * (angle / sinha);
    }
    else
    {
      r = qv * (static_cast<T>(2.0) / q_err.w());
    }
    
    return true;
  }

  // cost function generator--ONLY FOR PYTHON WRAPPER
  static ceres::CostFunction *Create(Eigen::Vector4d &x) {
    return new ceres::AutoDiffCostFunction<QuaterniondFunctor, 3, 4>(new QuaterniondFunctor(x));
  }

private:
  Eigen::Quaternion<double> q_;
  Eigen::Quaternion<double> q_inv_;
};

// AutoDiff local parameterization for the compact SE3 pose [t q] object. 
// The boxplus operator informs Ceres how the manifold evolves and also  
// allows for the calculation of derivatives.
struct cSE3Plus {
  // boxplus operator for both doubles and jets
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    // capture argument memory
    Sophus::SE3<T> X;
    cSE3convert(x, X);
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> dX(delta);
    Eigen::Map<Eigen::Matrix<T, 7, 1>>       Yvec(x_plus_delta);

    // increment pose using the exponential map
    Sophus::SE3<T> Exp_dX = Sophus::SE3<T>::exp(dX);
    Sophus::SE3<T> Y = X * Exp_dX;

    // assign SE3 coefficients to compact vector
    Eigen::Matrix<T,7,1> YvecCoef;
    cSE3convert(Y, YvecCoef);
    Yvec = YvecCoef;

    return true;
  }
  
  // local parameterization generator--ONLY FOR PYTHON WRAPPER
  static ceres::LocalParameterization *Create() {
    return new ceres::AutoDiffLocalParameterization<cSE3Plus,
                                                    7,
                                                    6>(new cSE3Plus);
  }
};

// AutoDiff local parameterization for the Transform3D object. The boxplus
// operator informs Ceres how the manifold evolves and also allows for the 
// calculation of derivatives
struct Transform3DPlus {
  // boxplus operator for both doubles and jets
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    // capture argument memory
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(x);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> dt(delta);
    const Eigen::Quaternion<T> q(x+3);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> dq(delta+3);
    Eigen::Map<Eigen::Matrix<T, 7, 1>> Tplus(x_plus_delta);

    // compute trivial vector addition component
    Tplus.template block<3,1>(0,0) = t + dt;

    // compute quaternion oplus component
    Eigen::Quaternion<T> plusq;
    T th = dq.norm();
    if (th > static_cast<T>(0.0))
    {
      Eigen::Matrix<T, 3, 1> u = dq / th;
      T sth2 = sin(th / static_cast<T>(2.0));
      plusq = Eigen::Quaternion<T>(cos(th / static_cast<T>(2.0)), u.x()*sth2, u.y()*sth2, u.z()*sth2);
    }
    else
    {
      Eigen::Matrix<T, 3, 1> r2 = dq / static_cast<T>(2.0);
      plusq = Eigen::Quaternion<T>(static_cast<T>(1.0), r2.x(), r2.y(), r2.z()).normalized();
    }
    Eigen::Quaternion<T> qplusq = q * plusq;
    Tplus.template block<4,1>(3,0) = qplusq.coeffs();

    return true;
  }

  // local parameterization generator--ONLY FOR PYTHON WRAPPER
  static ceres::LocalParameterization *Create() {
    return new ceres::AutoDiffLocalParameterization<Transform3DPlus,
                                                    7,
                                                    6>(new Transform3DPlus);
  }
};

void add_custom_cost_functions(py::module &m) {

  // Use pybind11 code to wrap your own cost function which is defined in C++

  // Transform3DLocalParameterization
  m.def("Transform3DLocalParameterization", &Transform3DPlus::Create);

  // cSE3LocalParameterization
  m.def("cSE3LocalParameterization", &cSE3Plus::Create);

  // cSE3Factor
  m.def("cSE3Factor", &cSE3Functor::Create);

  // RangeFactor
  m.def("RangeFactor", &RangeFunctor::Create);

  // Transform3DFactor
  m.def("Transform3DFactor", &Transform3DFunctor::Create);

  // Here is an example
  m.def("CreateCustomExampleCostFunction", &ExampleFunctor::Create);

  // QuaterniondFunctor
  m.def("CreateQuaterniondADCostFunction", &QuaterniondFunctor::Create);

}
