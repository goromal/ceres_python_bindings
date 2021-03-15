#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

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

void add_custom_cost_functions(py::module &m) {

  // Use pybind11 code to wrap your own cost function which is defined in C++s


  // Here is an example
  m.def("CreateCustomExampleCostFunction", &ExampleFunctor::Create);

  // QuaterniondFunctor
  m.def("CreateQuaterniondADCostFunction", &QuaterniondFunctor::Create);

}
