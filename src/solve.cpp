#include <iostream>
#include <limits>
#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <stdexcept>
#include <unsupported/Eigen/IterativeSolvers>
#include <unsupported/Eigen/SparseExtra> // For reading MatrixMarket files

#include "cxxopts.hpp"

struct Options {

  std::string A_filename;
  std::string b_filename;
  double tolerance;
  int iterations;
};

template <class T> bool run(Options opt) {

  Eigen::SparseMatrix<T> A;

  Eigen::loadMarket(A, opt.A_filename);
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  Vector b;
  Eigen::loadMarketVector(b, opt.b_filename);

  Eigen::BiCGSTABL<Eigen::SparseMatrix<T>, Eigen::DiagonalPreconditioner<T>>
      solver;

  Vector x = solver.compute(A).solve(b);

  double error = (A * x - b).norm() / b.norm();
  std::cout << "iterations:" << solver.iterations() << " error"
            << solver.error() << " real error:" << error << " success:"
            << std::string((solver.info() == Eigen::ComputationInfo::Success)
                               ? "true"
                               : "false")
            << std::endl;

    return (solver.info() == Eigen::ComputationInfo::Success);
}

int main(int argc, char *argv[]) {

  cxxopts::Options options("solve", "BICGSTABL debugging");

  options.add_options()("t,threads", "How many threads to use",
                        cxxopts::value<int>()->default_value("1"))(
      "s,tolerance", "Tolerance to which solvers should converge",
      cxxopts::value<double>()->default_value("1e-12"))(
      "i,iterations", "max number of iterations",
      cxxopts::value<int>()->default_value("1000"))(
      "A,SparseMatrix", "MM Format File for Sparse Matrix",
      cxxopts::value<std::string>())("b,Rightside", "MM Format File for Vector",
                                     cxxopts::value<std::string>())(
      "h,help", "Print usage");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  Options opt;

  int threads = result["threads"].as<int>();
  opt.tolerance = result["tolerance"].as<double>();
  opt.iterations = result["iterations"].as<int>();
  opt.A_filename = result["SparseMatrix"].as<std::string>();
  opt.b_filename = result["Rightside"].as<std::string>();

  std::cout << "Running profiler with " << threads << " threads\n"
            << "Required tolerance " << opt.tolerance << " num iterations "
            << opt.iterations << "\nA:" << opt.A_filename << "\nb"
            << opt.b_filename << std::endl;

  omp_set_num_threads(threads);

  int sym;
  bool complexMatrix;
  bool isvector;
  bool foundfile =
      Eigen::getMarketHeader(opt.A_filename, sym, complexMatrix, isvector);
  if (!foundfile) {
    throw std::runtime_error("File " + opt.A_filename + " not found.");
  }
  if (isvector) {
    throw std::runtime_error("File " + opt.A_filename +
                             " contains a vector not a matrix.");
  }
  bool complexVector;
  foundfile =
      Eigen::getMarketHeader(opt.b_filename, sym, complexVector, isvector);
  if (!foundfile) {
    throw std::runtime_error("File " + opt.b_filename + " not found.");
  }
  if (!isvector) {
    throw std::runtime_error("File " + opt.b_filename +
                             " contains a matrix not a vector.");
  }

  if (complexVector != complexMatrix) {
    throw std::runtime_error(
        "Both files have to have the same datatype complex or real");
  }
  bool error;
  if (complexMatrix) {
    error=run<std::complex<double>>(opt);
  } else {
    error=run<double>(opt);
  }
  return error;
}
