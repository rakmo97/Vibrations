
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_a4b104d1a8789e2943c847e75043b957 : public Expression
  {
     public:
       double l;
double w;
double h;
double A;


       dolfin_expression_a4b104d1a8789e2943c847e75043b957()
       {
            _value_shape.push_back(3);
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = 0.0;
          values[1] = x[0] >= 0.48*l && x[0] <= 52 && near(x[1],w,tol) && x[2] >= 0.3*h && x[2] <= 0.7*h? A : 0.0;
          values[2] = 0.0;

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "l") { l = _value; return; }          if (name == "w") { w = _value; return; }          if (name == "h") { h = _value; return; }          if (name == "A") { A = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "l") return l;          if (name == "w") return w;          if (name == "h") return h;          if (name == "A") return A;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_a4b104d1a8789e2943c847e75043b957()
{
  return new dolfin::dolfin_expression_a4b104d1a8789e2943c847e75043b957;
}

