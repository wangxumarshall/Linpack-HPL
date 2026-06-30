#!/bin/bash
#
# build_sdc_test.sh - Standalone build script for HPL SDC Fault Injection Test
#
# This script builds the SDC test program independently.
# Prerequisites:
#   - HPL library (libhpl.a) must be built first with -DHPL_SDC_CHECK -DHPL_SDC_INJECT
#   - MPI compiler (mpicc) must be available
#   - OpenBLAS or equivalent BLAS library
#
# Usage:
#   cd hpl/
#   bash testing/sdc_test/build_sdc_test.sh [arch]
#
# Default arch: WSL_OpenBLAS
#

ARCH=${1:-WSL_OpenBLAS}
TOPDIR="$(cd "$(dirname "$0")/.." && pwd)"
HPL_LIB="${TOPDIR}/lib/${ARCH}/libhpl.a"
TEST_SRC="${TOPDIR}/testing/sdc_test/HPL_sdc_test.c"
TEST_BIN="${TOPDIR}/bin/${ARCH}/xhpl_sdc_test"

# Compiler settings (override via environment)
CC=${CC:-mpicc}
CFLAGS="${CFLAGS:--O2 -w}"
SDC_FLAGS="-DHPL_SDC_CHECK -DHPL_SDC_INJECT"

# Include paths
INC="-I${TOPDIR}/include -I${TOPDIR}/include/${ARCH}"

# Libraries
LA_LIBS=${LA_LIBS:--lopenblas}

echo "============================================"
echo "  HPL SDC Fault Injection Test - Build"
echo "============================================"
echo "  ARCH:      ${ARCH}"
echo "  CC:        ${CC}"
echo "  HPL_LIB:   ${HPL_LIB}"
echo "  TEST_SRC:  ${TEST_SRC}"
echo "  TEST_BIN:  ${TEST_BIN}"
echo "============================================"

# Check prerequisites
if [ ! -f "${HPL_LIB}" ]; then
    echo ""
    echo "ERROR: HPL library not found at ${HPL_LIB}"
    echo "Please build HPL first:"
    echo "  cd ${TOPDIR}"
    echo "  make arch=${ARCH} setup"
    echo "  make arch=${ARCH} build"
    exit 1
fi

if [ ! -f "${TEST_SRC}" ]; then
    echo "ERROR: Test source not found at ${TEST_SRC}"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$(dirname "${TEST_BIN}")"

# Compile
echo ""
echo "Compiling..."
${CC} ${SDC_FLAGS} ${INC} ${CFLAGS} -o "${TEST_BIN}" "${TEST_SRC}" "${HPL_LIB}" ${LA_LIBS} -lm

if [ $? -eq 0 ]; then
    echo ""
    echo "BUILD SUCCESSFUL"
    echo "  Binary: ${TEST_BIN}"
    echo ""
    echo "Run with:"
    echo "  mpirun -np 4 ${TEST_BIN}"
    echo ""
else
    echo ""
    echo "BUILD FAILED"
    exit 1
fi
