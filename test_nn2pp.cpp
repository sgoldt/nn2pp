/*
 * File:    test_nn2pp.cpp
 *
 * Author:  Sebastian Goldt <goldt.sebastian@gmail.com>
 *
 * Date:    December 2018
 *
 * Tests for the Soft Committee code.
 */

#include <cmath>
#include <iostream>

#include <armadillo>

#include "gtest/gtest.h"

#include "libnn2pp.h"

using namespace arma;

class NN2Test : public ::testing::Test {
 public:
  NN2Test() {
  }

  ~NN2Test() {
  }
};


TEST_F(NN2Test, g_relu) {
  mat x = {{-1, -1.3, 3},
           {2, 1.3, 0.3},
           {-2.3, 4, 5}};

  mat g = {{0, 0, 3},
           {2, 1.3, 0.3},
           {0, 4, 5}};

  ASSERT_TRUE(approx_equal(g, g_relu(x), "absdiff", 0.000001)) \
      << "g(x) wrong for ReLU activation!";
}

TEST_F(NN2Test, g_relu_all_activations_neg) {
  mat x = mat(2, 3);
  x << -1 << -1.3 << -3 << endr
    << -2.3 << -4 << -5 << endr;

  mat g = mat(2, 3);
  g << 0 << 0 << 0 << endr
    << 0 << 0 << 0 << endr;

  ASSERT_TRUE(approx_equal(g, g_relu(x), "absdiff", 0.000001)) \
      << "g(x) wrong for ReLU activation if all activations are negative!";
}

TEST_F(NN2Test, dgdx_relu) {
  mat x = mat(3, 3);
  x << -1 << -1.3 << 3 << endr
    << 2 << 1.3 << 0.3 << endr
    << -2.3 << 4 << 5 << endr;

  mat dgdx = mat(3, 3);
  dgdx << 0 << 0 << 1 << endr
       << 1 << 1 << 1 << endr
       << 0 << 1 << 1 << endr;

  ASSERT_TRUE(approx_equal(dgdx, dgdx_relu(x), "absdiff", 0.000001)) \
      << "dgdx wrong for ReLU activation!";
}

TEST_F(NN2Test, phi_erf_singleLayer) {
  int bs = 2;
  int N = 3;
  int K = 2;
  
  mat w = randn<mat>(K, N);

  mat xis = randn<mat>(bs, N);

  mat phis_expected(bs, 1, fill::zeros);
  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      phis_expected(b, 0) += as_scalar(erf(w.row(k) * xis.row(b).t() / sqrt(N * 2)));
    }
  }
  phis_expected.print("expected=");

  // erf is the default function, so do not need to specify the activation 
  ASSERT_TRUE(approx_equal(phis_expected, phi(w, xis, g_erf), "absdiff", 0.000001)) << "phis are wrong!";
}

TEST_F(NN2Test, phi_erf) {
  int bs = 2;
  int N = 3;
  int K = 2;
  
  mat w = randn<mat>(K, N);
  vec v = randn<vec>(K);

  mat xis = randn<mat>(bs, N);

  mat phis_expected(bs, 1, fill::zeros);
  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      phis_expected(b, 0) += as_scalar(v(k) * erf(w.row(k) * xis.row(b).t() / sqrt(N * 2)));
    }
  }

  // erf is the default function, so do not need to specify the activation 
  ASSERT_TRUE(approx_equal(phis_expected, phi(w, v, xis, g_erf), "absdiff", 0.000001)) << "phis are wrong!";
}

TEST_F(NN2Test, phi_erf_normalised) {
  int bs = 2;
  int N = 3;
  int K = 2;
  
  mat w = randn<mat>(K, N);
  vec v = vec(K);
  v.fill(1. / K);
  
  mat xis = randn<mat>(bs, N);

  mat phis_expected(bs, 1, fill::zeros);
  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      phis_expected(b, 0) += as_scalar(erf(w.row(k) * xis.row(b).t() / sqrt(N * 2)));
    }
    phis_expected(b, 0) /= K;
  }

  // erf is the default function, so do not need to specify the activation 
  ASSERT_TRUE(approx_equal(phis_expected, phi(w, v, xis, g_erf),
                           "absdiff", 0.000001)) << "phis are wrong!";
}

TEST_F(NN2Test, phi_relu_normalised) {
  int bs = 2;
  int N = 3;
  int K = 2;
  
  mat w = randn<mat>(K, N);
  vec v = vec(K);
  v.fill(1. / K);
  
  mat xis = randn<mat>(bs, N);

  mat phis_expected(bs, 1, fill::zeros);
  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      double act = as_scalar(w.row(k) * xis.row(b).t() / sqrt(N));
      phis_expected(b, 0) += std::max(0., act);
    }
    phis_expected(b, 0) /= K;
  }

  mat (*g)(mat&) = g_relu;
  ASSERT_TRUE(approx_equal(phis_expected, phi(w, v, xis, g),
                           "absdiff", 0.000001)) << "phis are wrong!";
}

TEST_F(NN2Test, phi_relu_singleLayer) {
  int bs = 2;
  int N = 3;
  int K = 2;
  
  mat w = randn<mat>(K, N);
  
  mat xis = randn<mat>(bs, N);

  mat phis_expected(bs, 1, fill::zeros);
  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      double act = as_scalar(w.row(k) * xis.row(b).t() / sqrt(N));
      phis_expected(b, 0) += std::max(0., act);
    }
  }

  mat (*g)(mat&) = g_relu;
  ASSERT_TRUE(approx_equal(phis_expected, phi(w, xis, g), "absdiff", 0.000001))
      << "phis are wrong!";
}

TEST_F(NN2Test, phi_relu) {
  int bs = 2;
  int N = 3;
  int K = 2;
  
  mat w = randn<mat>(K, N);
  vec v = randn<vec>(K);
  
  mat xis = randn<mat>(bs, N);

  mat phis_expected(bs, 1, fill::zeros);
  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      double act = as_scalar(w.row(k) * xis.row(b).t() / sqrt(N));
      phis_expected(b, 0) += v(k) * std::max(0., act);
    }
  }

  mat (*g)(mat&) = g_relu;
  ASSERT_TRUE(approx_equal(phis_expected, phi(w, v, xis, g), "absdiff", 0.000001))
      << "phis are wrong!";
}

TEST_F(NN2Test, eg_erf_scm) {
  mat w;

  w << 0.34998185 << -0.51730095 << -1.27672365 <<  0.15857457 <<  0.80636752
    << -1.30266436 <<  1.04553608 << -1.40581276 << -0.40119664 <<  0.31838932 << endr 
    << 0.25096269 << -0.70390653 << -1.60709434 << -1.4129378  <<  0.29695629 
    << -0.22066656 << -0.06572677 << -0.59777609 << -0.83076419 << -0.17760066 << endr
    << 1.35757829 << -0.90800698 << -1.06598149 <<  0.93462077 <<  0.38120508 
    << -0.11103256 <<  0.20648807 << -0.14869054 <<  1.72580801 <<  0.59446499 << endr;
  vec v = vec(3, fill::ones);

  mat B;
  B << -0.82269399 << -0.36359333 << -0.24440212 <<  0.49828238 <<  0.61007546
    << 0.82557495 << -0.28769147 <<  0.24358059 << -1.31009873 << -1.15591452 << endr
    << -0.28590803 << -0.79115683 <<  0.40729374 <<  0.4940799  << -1.17793196
    << 0.56464427 <<  1.99950537 <<  0.64308283 <<  0.31500115 <<  0.18630809 << endr;
  vec A = vec(2, fill::ones);

  // result computed using Python implementation
  ASSERT_TRUE(abs(eg_analytical(B, A, w, v, g_erf, g_erf) - 1.08045) < 0.0001);
}

TEST_F(NN2Test, eg_erf_both) {
  mat B = {{-0.878181, -0.675171, 0.200794, -0.385201, -1.47387},
           {-0.0760653, -0.450636, 0.887182, 0.376101, -0.0873384}};
  vec A = {0.965846, -1.40304};
  mat w = {{1.04404, 0.583583, 1.43897, 0.963895, 0.795075},
           {-0.864295, -0.911849, 2.02883, 0.783554, -1.01488},
           {-0.451547, -0.772216, 0.780296, 0.799456, -0.0126815}};
  vec v = {-1.3598, -0.0671005, -0.395626};

  // result computed using Mathematica
  ASSERT_TRUE(abs(eg_analytical(B, A, w, v, g_erf, g_erf) - 0.0803271) < 0.0001);
}

TEST_F(NN2Test, eg_lin_both) {
  mat B = {{-0.878181, -0.675171, 0.200794, -0.385201, -1.47387},
           {-0.0760653, -0.450636, 0.887182, 0.376101, -0.0873384}};
  vec A = {0.965846, -1.40304};
  mat w = {{1.04404, 0.583583, 1.43897, 0.963895, 0.795075},
           {-0.864295, -0.911849, 2.02883, 0.783554, -1.01488},
           {-0.451547, -0.772216, 0.780296, 0.799456, -0.0126815}};
  vec v = {-1.3598, -0.0671005, -0.395626};

  // result computed using Mathematica
  ASSERT_TRUE(abs(eg_analytical(B, A, w, v, g_lin, g_lin) - 0.287911) < 0.0001);
}

TEST_F(NN2Test, gradientSingle_erf) {
  int N = 7;
  int K = 3;
  int bs = 2;

  mat w = randn<mat>(K, N);
  mat gradw = mat(size(w));
  mat xis = randn<mat>(bs, N);
  mat ys = randn<mat>(bs, 1);
  mat phis = phi(w, xis, g_erf);

  mat increment = mat(size(w), fill::zeros);

  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      mat act = w.row(k) * xis.row(b).t() / sqrt(N);
      increment.row(k) += 1.0 / bs * (ys(b, 0) - phis(b, 0)) * as_scalar(dgdx_erf(act)) * xis.row(b);
    }
  }

  update_gradient(gradw, w, xis, ys, g_erf, dgdx_erf);

  ASSERT_TRUE(approx_equal(increment, gradw,
                           "absdiff", 0.1)) << "step of sgd is wrong!";
}


TEST_F(NN2Test, gradient1_erf) {
  int N = 7;
  int K = 3;
  int bs = 2;

  mat w = randn<mat>(K, N);
  mat gradw = mat(size(w));
  vec v = randn<vec>(K);
  vec gradv = vec(size(v));
  mat xis = randn<mat>(bs, N);
  mat ys = randn<mat>(bs, 1);
  mat phis = phi(w, v, xis, g_erf);

  mat increment = mat(size(w), fill::zeros);

  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      mat act = w.row(k) * xis.row(b).t() / sqrt(N);
      increment.row(k) += 1.0 / bs * (ys(b, 0) - phis(b, 0)) * as_scalar(v(k) * dgdx_erf(act)) * xis.row(b);
    }
  }

  bool both = false;
  update_gradients(gradw, gradv, w, v, xis, ys, g_erf, dgdx_erf, both);

  ASSERT_TRUE(approx_equal(increment, gradw,
                           "absdiff", 0.1)) << "step of sgd is wrong!";
}

TEST_F(NN2Test, gradientSingle_relu) {
  int N = 7;
  int K = 3;
  int bs = 2;

  mat w = randn<mat>(K, N);
  mat gradw = mat(size(w));
  mat xis = randn<mat>(bs, N);
  mat ys = randn<mat>(bs, 1);
  mat phis = phi(w, xis, g_relu);

  mat increment = mat(size(w), fill::zeros);

  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      mat act = w.row(k) * xis.row(b).t() / sqrt(N);
      increment.row(k) += 1.0 / bs * (ys(b, 0) - phis(b, 0)) * as_scalar(dgdx_relu(act)) * xis.row(b);
    }
  }

  update_gradient(gradw, w, xis, ys, g_relu, dgdx_relu);

  ASSERT_TRUE(approx_equal(increment, gradw,
                           "absdiff", 0.1)) << "step of sgd is wrong!";
}

TEST_F(NN2Test, gradient1_relu) {
  int N = 7;
  int K = 3;
  int bs = 2;

  mat w = randn<mat>(K, N);
  mat gradw = mat(size(w));
  vec v = randn<vec>(K);
  vec gradv = vec(size(v));
  mat xis = randn<mat>(bs, N);
  mat ys = randn<mat>(bs, 1);
  mat phis = phi(w, v, xis, g_relu);

  mat increment = mat(size(w), fill::zeros);

  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      mat act = w.row(k) * xis.row(b).t() / sqrt(N);
      increment.row(k) += 1.0 / bs * (ys(b, 0) - phis(b, 0)) * as_scalar(v(k) * dgdx_relu(act)) * xis.row(b);
    }
  }

  bool both = false;
  update_gradients(gradw, gradv, w, v, xis, ys, g_relu, dgdx_relu, both);

  ASSERT_TRUE(approx_equal(increment, gradw,
                           "absdiff", 0.1)) << "step of sgd is wrong!";
}

TEST_F(NN2Test, gradient2_erf) {
  int N = 7;
  int K = 3;
  int bs = 10;

  mat w = randn<mat>(K, N);
  mat gradw = mat(size(w));
  vec v = randn<vec>(K);
  vec gradv = vec(size(v));
  mat xis = randn<mat>(bs, N);
  mat ys = randn<mat>(bs, 1);
  mat phis = phi(w, v, xis, g_erf);

  vec increment = vec(size(v), fill::zeros);

  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      mat act = w.row(k) * xis.row(b).t() / sqrt(N);
      increment(k) += 1.0 / bs * (ys(b, 0) - phis(b, 0)) * as_scalar(g_erf(act));
    }
  }

  update_gradients(gradw, gradv, w, v, xis, ys, g_erf, dgdx_erf, true);

  ASSERT_TRUE(approx_equal(increment, gradv,
                           "absdiff", 0.1)) << "step of sgd is wrong!";
}

TEST_F(NN2Test, gradient2_relu) {
  int N = 7;
  int K = 3;
  int bs = 10;

  mat w = randn<mat>(K, N);
  mat gradw = mat(size(w));
  vec v = randn<vec>(K);
  vec gradv = vec(size(v));
  mat xis = randn<mat>(bs, N);
  mat ys = randn<mat>(bs, 1);
  mat phis = phi(w, v, xis, g_relu);

  vec increment = vec(size(v), fill::zeros);

  for (int b = 0; b < bs; b++) {
    for (int k = 0; k < K; k++) {
      double act = as_scalar(w.row(k) * xis.row(b).t() / sqrt(N));
      increment(k) += 1.0 / bs * (ys(b, 0) - phis(b, 0)) * std::max(0., act);
    }
  }

  update_gradients(gradw, gradv, w, v, xis, ys, g_relu, dgdx_relu, true);

  ASSERT_TRUE(approx_equal(increment, gradv,
                           "absdiff", 0.1)) << "step of sgd is wrong!";
}


TEST_F(NN2Test, frac_error_polar) {
  // all correct
  mat a = {{-1., 1., 1., -1., -1., 1.}};
  mat b = {{-1., 1., 1., -1., -1., 1.}};
  ASSERT_DOUBLE_EQ(0, frac_error(a, b));

  // few wrong
  a = {{-1., 1., 1., -1., -1., 1.}};
  b = {{1., -1., 1., -1., -1., 1.}};
  ASSERT_DOUBLE_EQ(2. / a.n_elem, frac_error(a, b));

  // few correct
  a = {{-1., 1., 1., -1., -1., 1.}};
  b = {{1., -1., -1., 1., 1., 1.}};
  ASSERT_DOUBLE_EQ((a.n_elem - 1.) / a.n_elem, frac_error(a, b));

  // all wrong
  a = {{-1., 1., 1., -1., -1., 1.}};
  b = {{1., -1., -1., 1., 1., -1.}};
  ASSERT_DOUBLE_EQ(1, frac_error(a, b));
}

TEST_F(NN2Test, frac_error_relu_identical_networks) {
  int M = 4;
  int N = 784;
  mat B0 = randn<mat>(M, N);
  vec A0 = {1, 1, -1, -1};
  mat B = B0;
  vec A = A0;

  int NUM_TEST_SAMPLES = 50000;
  mat test_xs = randn<mat>(NUM_TEST_SAMPLES, N);
  mat test_ys = classify(B0, A0, test_xs, g_relu);
  mat test_preds = classify(B, A, test_xs, g_relu);

  ASSERT_DOUBLE_EQ(0., frac_error(test_ys, test_preds));
}


TEST_F(NN2Test, frac_error_relu) {
  int M = 4;
  int N = 784;
  mat B0 = randn<mat>(M, N);
  vec A0 = {1, 1, -1, -1};
  mat w = B0 + 1e-2 * randn<mat>(M, N);
  vec v = A0 + 1e-2 * randn<vec>(M);

  int NUM_TEST_SAMPLES = 50000;
  mat test_xs = randn<mat>(NUM_TEST_SAMPLES, N);
  mat test_ys = classify(B0, A0, test_xs, g_relu);
  mat test_preds = classify(w, v, test_xs, g_relu);

  double mse = mse_numerical(w, v, test_xs, test_ys, g_relu);
  cout << "mse=" << mse << endl;

  int num_errors = 0;
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    if ((as_scalar(test_ys(i, 0)) > 0 and as_scalar(test_preds(i, 0)) < 0)
        or (as_scalar(test_ys(i, 0)) < 0 and as_scalar(test_preds(i, 0)) > 0)) {
      num_errors++;
    }
  }
  double frac_error_expected = 1. * num_errors / NUM_TEST_SAMPLES;

  ASSERT_NEAR(frac_error_expected, frac_error(test_ys, test_preds), 1e-9);
}

TEST_F(NN2Test, frac_error_binary) {
  // all correct
  mat a = {{0., 1., 1., 0., 0., 1.}};
  mat b = {{0., 1., 1., 0., 0., 1.}};
  ASSERT_DOUBLE_EQ(0, frac_error(a, b));

  // few wrong
  a = {{0., 1., 1., 0., 0., 1.}};
  b = {{1., 0., 1., 0., 0., 1.}};
  ASSERT_DOUBLE_EQ(2. / a.n_elem, frac_error(a, b));

  // few correct
  a = {{0., 1., 1., 0., 0., 1.}};
  b = {{1., 0., 0., 1., 1., 1.}};
  ASSERT_DOUBLE_EQ((a.n_elem - 1.) / a.n_elem, frac_error(a, b));

  // few correct, encoding: 0, const rather than 0, 1
  a *= 4;
  b *= 5;
  ASSERT_DOUBLE_EQ((a.n_elem - 1.) / a.n_elem, frac_error(a, b));

  // all wrong
  a = {{0., 1., 1., 0., 0., 1.}};
  b = {{1., 0., 0., 1., 1., 0.}};
  ASSERT_DOUBLE_EQ(1, frac_error(a, b));
}

TEST_F(NN2Test, classify) {
  mat (*g)(mat&) = g_erf;
  int N = 15;
  int K = 3;
  int bs = 10;
  
  mat xis = randn<mat>(bs, N);
  mat w = randn<mat>(K, N);
  vec v = ones<vec>(K);
  mat phis = phi(w, v, xis, g);

  mat labels = mat(bs, 1, fill::zeros);
  for (int i = 0; i < bs; i++) {
    labels(i, 0) = (phis(i, 0) + 1e-9 > 0 ? 1 : -1);
  }

  ASSERT_TRUE(approx_equal(labels, classify(w, v, xis, g), "absdiff", 0.000001))
      << "classification wrong for g=erf!";  
}

TEST_F(NN2Test, classify_outputs) {
  int bs = 20;
  mat ys = randn<mat>(bs, 1);
  ys(0) = 1;

  mat labels = sign(ys);
  labels(0) = 1;

  ASSERT_TRUE(approx_equal(labels, classify(ys), "absdiff", 0.000001))
      << "classification of network outputs wrong for g=erf!";  
}

TEST_F(NN2Test, init_student_randomly_2ndlayer) {
  double uniform_default = 0;
  double both_default = false;
  double normalise_default = false;
  double meanfield_default = false;
  
  int N = 15;
  int K = 3;
  mat w = mat(K, N);
  vec v = vec(K);
  double uniform = datum::pi;

  // possible results for the second layer
  vec vecOnes = ones<vec>(K);
  vec vecOnesOverK = 1. / K * ones<vec>(K);
  vec vecOnesOverSqrtK = 1. / sqrt(K) * ones<vec>(K);
  vec vecUniform = uniform * ones<vec>(K);
  vec vecUniformOverK = uniform / K * ones<vec>(K);
  vec vecUniformOverSqrtK = uniform / sqrt(K) * ones<vec>(K);

  for (int init = 1; init <= 2; init++) {
    // no options given
    init_student_randomly(w, v, N, K, init, uniform_default, both_default,
                  normalise_default, meanfield_default);
    ASSERT_TRUE(approx_equal(vecOnes, v, "absdiff", 0.000001)) <<
        "initialisation wrong with default options";

    // --both: random v
    init_student_randomly(w, v, N, K, init,
                  uniform_default, true, normalise_default, meanfield_default);
    ASSERT_FALSE(abs(v(0) - v(1)) < 0.000001) <<
        "initialisation wrong with --both";

    // --uniform: v = A
    init_student_randomly(w, v, N, K, init,
                  uniform, both_default, normalise_default, meanfield_default);
    ASSERT_TRUE(approx_equal(vecUniform, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --uniform";

    // --both --uniform: random v
    init_student_randomly(w, v, N, K, init,
                  uniform_default, true, normalise_default, meanfield_default);
    ASSERT_FALSE(approx_equal(vecUniform, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --both --uniform";
    ASSERT_FALSE(abs(v(0) - v(1)) < 0.000001) <<
        "initialisation wrong with --both --uniform";

    // NOW THE SAME THING AGAIN with --normalise
    // --normalise: v = 1 /K
    init_student_randomly(w, v, N, K, init,
                  uniform_default, both_default, true, meanfield_default);
    ASSERT_TRUE(approx_equal(vecOnesOverK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --normalise";

    // --both --normalise v = random (both overrides normalise)
    init_student_randomly(w, v, N, K, init,
                  uniform_default, true, true, meanfield_default);
    ASSERT_FALSE(approx_equal(vecOnesOverK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --both --uniform";
    ASSERT_FALSE(abs(v(0) - v(1)) < 0.000001) <<
        "initialisation wrong with --both --uniform";

    // --uniform --normalise
    init_student_randomly(w, v, N, K, init,
                  uniform, both_default, true, meanfield_default);
    ASSERT_TRUE(approx_equal(vecOnesOverK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --uniform --normalise";

    // --both --uniform --normalise v = random
    init_student_randomly(w, v, N, K, init,
                  uniform, true, true, meanfield_default);
    ASSERT_FALSE(approx_equal(vecUniformOverK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --both --uniform";
    ASSERT_FALSE(abs(v(0) - v(1)) < 0.000001) <<
        "initialisation wrong with --both --uniform";

    // NOW THE SAME THING AGAIN with --meanfield
    // --meanfield: v = 1 / sqrt(K)
    init_student_randomly(w, v, N, K, init,
                  uniform_default, both_default, false, true);
    ASSERT_TRUE(approx_equal(vecOnesOverSqrtK, v, "absdiff", 0.1)) <<
        "initialisation wrong with --meanfield";

    // --both --meanfield  v = random (both overrides normalise for student)
    init_student_randomly(w, v, N, K, init,
                  uniform_default, true, false, true);
    ASSERT_FALSE(approx_equal(vecOnesOverK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --both --meanfield";
    ASSERT_FALSE(approx_equal(vecOnesOverSqrtK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --both --meanfield";
    ASSERT_FALSE(abs(v(0) - v(1)) < 0.000001) <<
        "initialisation wrong with --both --meanfield";

    // --uniform --meanfield
    init_student_randomly(w, v, N, K, init,
                  uniform, both_default, false, true);
    ASSERT_TRUE(approx_equal(vecUniformOverSqrtK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --uniform --meanfield";

    // --both --uniform --meanfield v = random
    init_student_randomly(w, v, N, K, init,
                  uniform, true, false, true);
    ASSERT_FALSE(approx_equal(vecUniformOverK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --both --uniform --meanfield";
    ASSERT_FALSE(approx_equal(vecUniformOverSqrtK, v, "absdiff", 0.000001)) <<
        "initialisation wrong with --both --uniform --meanfield";
    ASSERT_FALSE(abs(v(0) - v(1)) < 0.000001) <<
        "initialisation wrong with --both --uniform --meanfield";
  }
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // set the seed
  arma_rng::set_seed(1234);

  return RUN_ALL_TESTS();
}
