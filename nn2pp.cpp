/*
 * File:    nn2pp.cpp
 *
 * Author:  Sebastian Goldt <goldt.sebastian@gmail.com>
 *
 * Version: 0.1
 *
 * Date:    December 2018
 */

#include <cmath>
#include <getopt.h>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

// #define ARMA_NO_DEBUG
#include <armadillo>
#include <chrono>
using namespace arma;

#include "libnn2pp.h"

const char * usage = R"USAGE(
This is nn2pp, a tool to analyse scalar neural networks with a single hidden
  layer in a teacher-student setup.

usage: nn2pp.exe [-h] [--g G] [-N N] [-M M] [-K K] [--lr LR] [--lr2 LR2] [--both]
                     [--sigma SIGMA] [--wd WD] [--bs BS] [--ts TS] 
                     [--init INIT] [--steps STEPS] [--epochs EPOCHS] [--numeric]
                     [--batch] [--normalise][--uniform A] [-s SEED] [--quiet]

optional arguments:
  -h, -?                show this help message and exit
  --g1 G1               activation function for the teacher;
                           0-> linear, 1->erf, 2->ReLU.
  --g2 G2               activation function for the student;
                           0-> linear, 1->erf, 2->ReLU.
  -N, --N N             input dimension
  -M, --M M             number of hidden units in the teacher network
  -K, --K K             number of hidden units in the student network
  -l, --lr LR           learning rate
  --lr2 LR2             learning rate for the second layer only. If not
                          specified, we will use the same learning rate for
                          both layers.
  -s, --sigma SIGMA     std. dev. of teacher's output noise. Default=0.
                          For classification, the probability that a label is
                          drawn at random.
  --sparse S            Hides a fraction S of entries of first-layer teacher
                          weights.
  -w, --wd WD           weight decay constant. Default=0.
  --bs BS               mini-batch size for SGD step. Default=1.
  --ts TS               For online learning from a fixed training set, this is
                          the training set's size in multiples of N. Default=0.
                          (corresponding to online learning)
  -a, --steps STEPS     max. weight update steps in multiples of N
  -e, --epochs EPOCHS   number of training epochs. Overrides steps.
  --both                train both layers.
  --uniform A           make all of the teacher's second layer weights equal to
                          this value. If the second layer of the student is not
                          trained, the second-layer output weights of the student
                          are also set to this value.
  -i, --init INIT       weight initialisation:
                          1,2: i.i.d. Gaussians with sd 1 or 1/sqrt(N), resp.
                            3: informed initialisation; only for K \ge M.
                            4: denoising initialisation
                            5: 'mixed': i.i.d. Gaussian with 1/sqrt(N), 1/sqrt(K)
  --stop                generalisation error at which to stop the simulation.
  --store               store initial overlap and final weight matrices.
  -z, --teacher PREFIX  load weights for teacher and student from files with the
                          given prefix.
  --numeric             calculate the generalisation error numerically.
  --batch               batch gradient descent (overrides --bs).
  --normalise           divides 2nd layer weights by M and K for teacher and
                          student, resp. Overwritten by --both for the student
                          (2nd layer weights of the student are initialised
                           according to --init in that case).
  --meanfield           divides 2nd layer weights by sqrt(M) and sqrt(K) for
                          teacher and student, resp. Overwritten by --both for
                          the student (2nd layer weights of the student are
                          initialised according to --init in that case).
  --mix                 changes the sign of half of the teacher's second-layer
                          weights.
  --classify            compute fractional test/training errors, too.
  -r SEED, --seed SEED  random number generator seed. Default=0
  --quiet               be quiet and don't print order parameters to cout.
)USAGE";

int main(int argc, char* argv[]) {
  // flags; false=0 and true=1
  int batch = 0; // perform batch gradient descent?
  int numeric = 0;  // calculate the generalisation error numerically
  int normalise = 0;  // normalise SCM outputs
  int meanfield = 0;  // 2nd layer = 1/sqrt(M)
  int mix = 0;  // change the sign of half of teacher's second-layer weights
  int classification = 0;  // consider a classification task
  int mnist = 0;  // use MNIST images as inputs
  int both = 0; // train both layers
  int store = 0; // store initial weights etc.
  int quiet = 0;  // don't print the order parameters to cout
  // other parameters
  int    g1        = ERF;  // teacher activation function
  int    g2        = ERF;  // student activation function
  int    N         = 784;  // number of inputs
  int    M         = 4;  // num of teacher's hidden units
  int    K         = 4;  // num of student's hidden units
  double lr        = 0.5;  // learning rate
  double lr2       = -1;  // learning rate for the second layer.
  double wd        = 0;  // weigtht decay constant
  double sigma     = 0;  // std.dev. of the teacher's additive output noise
  int    init      = INIT_LARGE;  // initial weights
  double stop      = 0;  // value of eg at which to stop the simulation
  double ts        = 0;  // set of the training set (opt.)
  int    bs        = 1;  // mini-batch size
  double max_steps = 1000;  // max number of gradient updates / N
  int    epochs    = 0;  // can give the simulation length if there is a finite
                         // training set; if provided, overrides max_steps
  int    seed      = 0;  // random number generator seed
  double sparse    = -1; // hide this fraction of teacher weights
  double uniform   = 0; // value of all weights in the teacher's second layer
  std::string teacher;       // name of file containing teacher weights
  char *train_xs_fname = NULL;  // name of file from which training inputs are read
  char *train_ys_fname = NULL;  // name of file from which training labels are read
  char *test_xs_fname = NULL;  // name of file from which test inputs are read
  char *test_ys_fname = NULL;  // name of file from which test labels are read

  // parse command line options using getopt
  int c;

  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"batch",      no_argument, &batch,          1},
    {"numeric",    no_argument, &numeric,        1},
    {"normalise",  no_argument, &normalise,      1},
    {"meanfield",  no_argument, &meanfield,      1},
    {"mix",        no_argument, &mix,            1},
    {"classify",   no_argument, &classification, 1},
    {"mnist",      no_argument, &mnist,          1},
    {"both",       no_argument, &both,           1},
    {"store",      no_argument, &store,          1},
    {"quiet",      no_argument, &quiet,          1},
    {"trainxs",  required_argument, 0, 'x'},
    {"trainys",  required_argument, 0, 'y'},
    {"testxs",  required_argument, 0, 'c'},
    {"testys",  required_argument, 0, 'd'},
    {"teacher", required_argument, 0, 'z'},
    {"g1",      required_argument, 0, 'f'},
    {"g2",      required_argument, 0, 'g'},
    {"N",       required_argument, 0, 'N'},
    {"M",       required_argument, 0, 'M'},
    {"K",       required_argument, 0, 'K'},
    {"lr",      required_argument, 0, 'l'},
    {"lr2",     required_argument, 0, 'm'},
    {"sigma",   required_argument, 0, 's'},
    {"wd",      required_argument, 0, 'w'},
    {"sparse",  required_argument, 0, 'p'},
    {"init",    required_argument, 0, 'i'},
    {"uniform", required_argument, 0, 'u'},
    {"stop",    required_argument, 0, 'j'},
    {"bs",      required_argument, 0, 'b'},
    {"ts",      required_argument, 0, 't'},
    {"steps",   required_argument, 0, 'a'},
    {"epochs",  required_argument, 0, 'e'},
    {"seed",    required_argument, 0, 'r'},
    {0, 0, 0, 0}
  };

  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "f:g:N:M:K:l:s:y:w:x:i:b:t:a:e:r:u:j:",
                    long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) {
      break;
    }

    switch (c) {
      case 0:
        break;
      case 'x':
        train_xs_fname = optarg;
        break;
      case 'y':
        train_ys_fname = optarg;
        break;
      case 'c':
        test_xs_fname = optarg;
        break;
      case 'd':
        test_ys_fname = optarg;
        break;
      case 'f':
        g1 = atoi(optarg);
        break;
      case 'g':
        g2 = atoi(optarg);
        break;
      case 'N':
        N = atoi(optarg);
        break;
      case 'M':
        M = atoi(optarg);
        break;
      case 'K':
        K = atoi(optarg);
        break;
      case 'l':
        lr = atof(optarg);
        break;
      case 'm':
        lr2 = atof(optarg);
        break;
      case 's':
        sigma = atof(optarg);
        break;
      case 'w':
        wd = atof(optarg);
        break;
      case 'p':  // sparse teacher
        sparse = atof(optarg);
        break;
      case 'u':  // value of the second layer weights of the teacher
        uniform = atof(optarg);
        break;
      case 'z':  // pre-load teacher weights from file with this prefix
        teacher = std::string(optarg);
        break;
      case 'b':  // mini-batch size
        bs = atoi(optarg);
        break;
      case 'i':  // initialisation of the weights
        init = atoi(optarg);
        break;
      case 'j':  // value of the second layer weights of the teacher
        stop = atof(optarg);
        break;
      case 't':  // size of the training set
        ts = atof(optarg);
        break;
      case 'a':  // number of steps in multiples of N
        max_steps = atof(optarg);
        break;
      case 'e':  // training epochs
        epochs = atoi(optarg);
        break;
      case 'r':  // random number generator seed
        seed = atoi(optarg);
        break;
      case 'h':  // intentional fall-through
      case '?':
        cout << usage << endl;
        return 0;
      default:
        abort ();
    }
    
  }
  // if not explicitly given, use the same learning rate in both layers
  if (lr2 < 0) {
    lr2 = lr;
  }

  if (meanfield and normalise) {
    cerr << "Cannot have both meanfield and normalised networks. Will exit now !" << endl;
    return 1;
  }

  // set the seed
  arma_rng::set_seed(seed);

  // Draw the weights of the network and their activation functions
  mat B0 = mat();  // teacher input-to-hidden weights
  vec A0 = vec();  // teacher hidden-to-output weights
  bool success = false;
  if (!teacher.empty()) {
    teacher.append("_w.dat");
    success = B0.load(teacher);
    teacher.replace(teacher.end()-6, teacher.end(), "_v.dat");
    success = success && A0.load(teacher);
    M = B0.n_rows;
    N = B0.n_cols;
  } else {
    success = init_teacher_randomly(B0, A0, N, M, uniform, both, normalise,
                                         meanfield, mix, sparse);
  }

  if (!success) {
    // some error happened during teacher init
    cerr << "Could not initialise teacher; will exit now!" << endl;
    return 1;
  }

  mat w = mat(K, N);   // student weights
  vec v = vec(K);

  switch (init) {
    case INIT_LARGE:
    case INIT_SMALL:
    case INIT_MIXED:
    case INIT_MIXED_NORMALISE: // intentional fall-through
      init_student_randomly(w, v, N, K, init, uniform, both, normalise, meanfield);
      break;
    case INIT_INFORMED:
      if (K < M) {
        cerr << "Cannot do an informed initialisation with K<M." << endl
             << "Will exit now !" << endl;
        return 1;
      } else {
        w = 1e-9 * randn<mat>(K, N);
        w.rows(0, M - 1) += B0;
        if (both) {
          v = 1e-9 * randn<vec>(K);
          v.head(M) += A0;
        } else {
          if (abs(uniform) > 0) {
            v = vec(K, fill::ones);
            v *= uniform;
          } else {
            v = vec(K, fill::ones);
          }
        }
      }
      break;
    case INIT_DENOISE:
      if (K < M) {
        cerr << "Cannot do a denoiser initialisation with K<M." << endl
             << "Will exit now !" << endl;
        return 1;
      }
      if (!both) {
        cerr << "Need to be able to change the second-layer weights to do a "
             << "denoiser initialisation. Will exit now !" << endl;
        return 1;
      }
      if (g1 == LINEAR and g2 == LINEAR) {
        // // works for M=2:
        // mat B_perceptron = A0.t() * B0 / M;
        // w.each_row() = B_perceptron / sqrt(K);
        // v.fill(1. * M / sqrt(K));

        mat B_perceptron = A0.t() * B0 / sqrt(M);
        w.each_row() = B_perceptron / sqrt(K);
        v.fill(1. * sqrt(M) / sqrt(K));
        // mat Q = w * w.t() / N;
        // mat R = w * B0.t() / N;

        // Q.print("Q=");
        // R.print("R=");
        // v.print("v=");

        // mat test_xs = randn<mat>(100000, N);
        // mat test_ys = phi(B0, A0, test_xs, g_lin);
        // double eg = mse_numerical(w, v, test_xs, test_ys, g_lin);
        // cout << "eg = " << eg << endl;
        // return 0;
      } else {
        for (int k = 0; k < K; k++) {
          w.row(k) = B0.row(k % M);
          v(k) = A0(k % M);
          // now do the proper rescaling:
          v(k) = (k % M) <= (K % M - 1) ? v(k)/(floor(K/M) + 1) : v(k)/floor(K/M);
        }
      }
      break;
    case INIT_NATI: {
      w = 1e-3 * randn<mat>(size(w));
      v = 1. / K * ones<vec>(K);
      break;
    }
    case INIT_NATI_MF: {
      w = 1e-3 * randn<mat>(size(w));
      v = 1. / sqrt(K) * ones<vec>(K);
      break;
    }
    default:
      cerr << "Init must be within 1-8. Will exit now." << endl;
      return 1;
  }

  mat (*g1_fun)(mat&);
  mat (*g2_fun)(mat&);
  mat (*dgdx_fun)(mat&);
  switch (g1) {  // find the teacher's activation function
    case LINEAR:
      g1_fun = g_lin;
      break;
    case ERF:
      g1_fun = g_erf;
      break;
    case RELU:
      g1_fun = g_relu;
      break;
    case QUAD:
      g1_fun = g_quad;
      break;
    default:
      cerr << "g1 has to be linear (g1=" << LINEAR << "), erg1 (g1=" << ERF <<
          "), ReLU (g1=" << RELU << ") or sign (g1=" << SIGN << ") or quad (g1="
           << QUAD << "). " << endl;
      cerr << "Will exit now!" << endl;
      return 1;
  }
  switch (g2) {  // find the teacher's activation function
    case LINEAR:
      g2_fun = g_lin;
      dgdx_fun = dgdx_lin;
      break;
    case ERF:
      g2_fun = g_erf;
      dgdx_fun = dgdx_erf;
      break;
    case RELU:
      g2_fun = g_relu;
      dgdx_fun = dgdx_relu;
      break;
    case QUAD:
      g2_fun = g_quad;
      dgdx_fun = dgdx_quad;
      break;
    default:
      cerr << "g1 has to be linear (g1=" << LINEAR << "), erg1 (g1=" << ERF <<
          "), ReLU (g1=" << RELU << ") or sign (g1=" << SIGN << ") or quad (g1="
           << QUAD << "). " << endl;
      cerr << "Will exit now!" << endl;
      return 1;
  }
  const char* g1_name = activation_name(g1);
  const char* g2_name = activation_name(g2);

  if (classification && sigma > 1) {
    cerr << "For classification, sigma has to be between 0 and 1." << endl
         << "Will exit now!" << endl;
    return 1;
  }

  std::ostringstream welcome;
  welcome << "# This is scm++" << endl
          << "# g1=" << g1_name << ", g2=" << g2_name
          << ", N=" << N << ", M=" << M << ", K=" << K
          << ", steps/N=" << max_steps << ", seed=" << seed << endl
          << "# lr=" << lr << ", lr2=" << lr2 << ", sigma=" << sigma
          << ", wd=" << wd << ", mini-batch size=" << bs << endl;
  if (!teacher.empty()) {
    welcome << "# Loaded teacher weights from " << teacher.c_str() << endl;
  }
  if (both) {
    welcome << "# training both layers";
    if (uniform > 0)
      welcome << " (teacher's second layer has uniform weights=" << uniform << ")";
    welcome << endl;
  }
  
  // generate a finite training set?
  mat train_xs = mat();
  mat train_ys = mat();
  // if this is a classification task and if g2=ReLU, need to know the
  // boundary between the two classes:
  if (ts > 0) {
    train_xs = randn<mat>((int) round(ts * N), N);
    train_ys = phi(B0, A0, train_xs, g1_fun);
    if (classification) {
      train_ys = classify(train_ys);
      if (sigma > 0) {
        randomise(train_ys, sigma);  // flip some of the outputs
      }
    } else if (sigma > 0) {
      train_ys += sigma * randn<mat>(size(train_ys));
    }
    welcome << "# Generated finite training set of size ts=" << ts << endl;
  }
  // load external training inputs?
  if (train_xs_fname != NULL) {
    bool loaded = train_xs.load(train_xs_fname, csv_ascii);
    if (!loaded) {
      cerr << "Problem loading training inputs from " << train_xs_fname << endl;
      return 1;
    }
    N = train_xs.n_cols;
    ts = round(train_xs.n_rows / N);
    welcome << "# Loaded training inputs from " << train_xs_fname << " of shape " << size(train_xs) << ", mean=" << mean(mean(train_xs)) << ", var=" << var(vectorise(train_xs)) << endl;

    // load external training labels or generate synthetic labels?
    if (train_ys_fname != NULL) {
      bool loaded = train_ys.load(train_ys_fname, csv_ascii);
      if (!loaded) {
        cerr << "Problem loading training labels from " << train_ys_fname << endl;
        return 1;
      }
      welcome << "# Loaded training labels from " << train_ys_fname << " of shape " << size(train_ys) << ", mean=" << mean(mean(train_ys)) << ", var=" << var(vectorise(train_ys)) << endl;
      M = 0;
      B0 = mat();
      A0 = vec();
    } else {
      train_ys = phi(B0, A0, train_xs, g1_fun);
      welcome << "# Generated training labels using teacher." << endl;
    }
  }

  // is this batch learning or SGD with mini-batches of a given size?
  if (batch) {
    if (ts == 0 && train_xs_fname == NULL) {
      cerr << "External training set of synthetic training set via --ts required for batch gradient descent." << endl;
      return 1;
    } 
    bs = train_xs.n_rows;
  }

  // how long do we train for?
  if (epochs > 0) {
    if (! (ts > 0)) {
      cerr << "Can only specify the number of training epochs if a fixed training set is given" << endl;
      return 1;
    }

    max_steps = epochs * ts / bs;
  }

  // find printing times
  vec steps = logspace<vec>(-1, log10(max_steps), 200);

  // generate a finite test set?
  numeric = ((g1 == RELU) || (g2 == RELU) || (g1 == QUAD) || (g2 == QUAD)
             || numeric || classification);
  mat test_xs;
  mat test_ys;
  if (test_xs_fname != NULL) {
    bool loaded = test_xs.load(test_xs_fname, csv_ascii);
    if (!loaded) {
      cerr << "Problem loading test inputs from " << test_xs_fname << endl;
      return 1;
    }
    welcome << "# Loaded test inputs from " << test_xs_fname << " of shape " << size(test_xs) << ", mean=" << mean(mean(test_xs)) << ", var=" << var(vectorise(test_xs)) << endl;

    // load external testing labels or generate synthetic labels?
    if (test_ys_fname != NULL) {
      bool loaded = test_ys.load(test_ys_fname, csv_ascii);
      if (!loaded) {
        cerr << "Problem loading test labels from " << test_ys_fname << endl;
        return 1;
      }
      welcome << "# Loaded testing labels data from " << test_ys_fname << " of shape " << size(test_ys) << ", mean=" << mean(mean(test_ys)) << ", var=" << var(vectorise(test_ys)) << endl;
    } else {
      welcome << "# Generated test labels using teacher." << endl;
      test_ys = phi(B0, A0, test_xs, g1_fun);
    }
  } else if (numeric || classification) {
    test_xs = randn<mat>(100000, N);
    test_ys = phi(B0, A0, test_xs, g1_fun);
    // we are comparing to the noiseless teacher output, so no noise is addded!
    if (classification) {
      test_ys = classify(test_ys);
    }
    welcome << "# Generated test set with 100000 samples" << endl;
  } else {
    test_xs.reset();
    test_ys.reset();
  }

  switch (init) {
    case INIT_SMALL:
      welcome << "# initial weights have small std dev" << endl;
      break;
    case INIT_MIXED:
      welcome << "# initial weights have mixed std dev 1/sqrt(N), 1/sqrt(K)" << endl;
      break;
    case INIT_LARGE:
      welcome << "# initial weights have std dev 1" << endl;
      break;      
    case INIT_INFORMED:
      welcome << "# informed initialisation" << endl;
      break;
  }
  welcome << "# steps / N, eg, et, eg_frac, et_frac, diff" << endl;
  std::string welcome_string = welcome.str();
  cout << welcome_string;

  if (test_ys_fname != NULL) {
    g1_name = "ext";
  }
  char* lr2_desc;
  asprintf(&lr2_desc, "2lr%g_", lr2);
  char* ts_desc;
  asprintf(&ts_desc, "ts%g_", ts);
  char* uniform_desc;
  asprintf(&uniform_desc, "u%g_", uniform);
  char* sparse_desc;
  asprintf(&sparse_desc, "sparse%g_", sparse);
  char* log_fname;
  asprintf(&log_fname,
           "nn2pp_%s_%s_%s_%s%s%s%s%s%s%sN%d_M%d_K%d_lr%g_%swd%g_sigma%g_bs%d_i%d_%ssteps%g_s%d.dat",
           g1_name, g2_name, (both ? "both" : "1st"),
           (uniform > 0 ? uniform_desc : ""), (mix > 0 ? "mix_" : ""),
           (normalise ? "norm_" : ""), (meanfield ? "mf_" : ""),
           (mnist ? "mnist_" : ""),
           (classification ? "class_" : ""),  (sparse > 0 ? sparse_desc : ""),
           N, M, K, lr, (lr2 != lr ? lr2_desc : ""), wd, sigma, bs, init,
           (ts > 0 ? ts_desc : ""), max_steps, seed);
  FILE* logfile = fopen(log_fname, "w");
  fprintf(logfile, "%s", welcome_string.c_str());

  // save initial conditions
  if (store) {
    mat Q0 = w * w.t() / N;

    std::string fname = std::string(log_fname);
    fname.replace(fname.end()-4, fname.end(), "_Q0.dat");
    Q0.save(fname, csv_ascii);
    if (!B0.is_empty()) {
      mat R0 = B0.is_empty() ? mat() : w * B0.t() / N;
      mat T0 = B0.is_empty() ? mat() : B0 * B0.t() / N;

      fname.replace(fname.end()-7, fname.end(), "_R0.dat");
      R0.save(fname, csv_ascii);
      fname.replace(fname.end()-7, fname.end(), "_T0.dat");
      T0.save(fname, csv_ascii);
      fname.replace(fname.end()-7, fname.end(), "_A0.dat");
      A0.save(fname, csv_ascii);
    }
    fname.replace(fname.end()-7, fname.end(), "_v0.dat");
    v.save(fname, csv_ascii);
  }

  std::clock_t c_start = std::clock();
  auto t_start = std::chrono::high_resolution_clock::now();
  learn(B0, A0, w, v, g1_fun, g2_fun, dgdx_fun,
        lr, lr2, wd, sigma, steps, bs,
        train_xs, train_ys, test_xs, test_ys, logfile,
        both, classification, quiet, 0, store, log_fname, stop);
  std::clock_t c_end = std::clock();
  auto t_end = std::chrono::high_resolution_clock::now();
  std::ostringstream time_stream;
  time_stream << "# CPU time used: "
             << (c_end-c_start) / CLOCKS_PER_SEC << " s\n"
              << "# Wall clock time passed: "
             << std::chrono::duration_cast<std::chrono::seconds>(t_end-t_start).count()
              << " s\n";
  std::string time_string = time_stream.str();
  cout << time_string;
  fprintf(logfile, "%s", time_string.c_str());
  fclose(logfile);

  if (store) {    // store the final teacher/student weights
    std::string fname = std::string(log_fname);
    fname.replace(fname.end()-4, fname.end(), "_w.dat");
    w.save(fname.c_str(), csv_ascii);
    fname.replace(fname.end()-6, fname.end(), "_v.dat");
    v.save(fname.c_str(), csv_ascii);
  }
  return 0;
}
