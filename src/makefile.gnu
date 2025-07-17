#!/bin/make -f

###############################################################################
# EXTERNAL LIBRARY PATHS
###############################################################################
# Enter here paths to GSL or EIGEN if they are not in your standard include
# path. DO NOT completely remove the entry, leave at least "./".
PROJECT_GSL=/opt/homebrew/Cellar/gsl/2.7.1/include
#PROJECT_GSL=${HOME}/local/src/gsl-2.8/build/include
PROJECT_EIGEN=/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/
#PROJECT_EIGEN=${HOME}/local/src/eigen/
PROJECT_BOOST=${BOOST_ROOT}

###############################################################################
# COMPILERS AND FLAGS
###############################################################################
PROJECT_CC=g++
PROJECT_MPICC=mpic++
# OpenMP parallelization is disabled by default, add flag "-fopenmp" to enable.
PROJECT_CFLAGS=-O3 -march=native -std=c++11 #-fopenmp
PROJECT_CFLAGS_MPI=-Wno-long-long
PROJECT_DEBUG=-g -pedantic-errors -Wall -Wextra
PROJECT_TEST=--coverage -fno-default-inline -fno-inline -fno-inline-small-functions -fno-elide-constructors
PROJECT_AR=ar
PROJECT_ARFLAGS=-rcsv
PROJECT_CFLAGS_BLAS=
PROJECT_LDFLAGS_BLAS=-L/opt/homebrew/Cellar/gsl/2.7.1/lib -L/opt/homebrew/Cellar/openblas/0.3.28/lib -lopenblas -lgsl -lgslcblas

###############################################################################
# COMPILE-TIME OPTIONS
###############################################################################

# Do not use symmetry function groups.
#PROJECT_OPTIONS+= -DN2P2_NO_SF_GROUPS

# Do not use symmetry function cache.
#PROJECT_OPTIONS+= -DN2P2_NO_SF_CACHE

# Disable asymmetric polynomial symmetry functions.
#PROJECT_OPTIONS+= -DN2P2_NO_ASYM_POLY

# Build with dummy Stopwatch class.
#PROJECT_OPTIONS+= -DN2P2_NO_TIME

# Disable check for low number of neighbors.
#PROJECT_OPTIONS+= -DN2P2_NO_NEIGH_CHECK

# Use alternative (older) memory layout for symmetry function derivatives.
#PROJECT_OPTIONS+= -DN2P2_FULL_SFD_MEMORY

# Compile without MPI support.
#PROJECT_OPTIONS+= -DN2P2_NO_MPI

# Use BLAS together with Eigen.
#PROJECT_OPTIONS+= -DEIGEN_USE_BLAS

# Disable all C++ asserts (also Eigen debugging).
#PROJECT_OPTIONS+= -DNDEBUG

# Use Intel MKL together with Eigen.
#PROJECT_OPTIONS+= -DEIGEN_USE_MKL_ALL

# Disable Eigen multi threading.
PROJECT_OPTIONS+= -DEIGEN_DONT_PARALLELIZE
