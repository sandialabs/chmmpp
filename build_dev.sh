#!/bin/bash -e
#
# This scripts builds chmmpp in the `build` directory to support local
# development and debugging.
#
# This uses Spack to install third-party dependencies in the `spack` directory.
#
spack_dev=0
spack_reinstall=0
coek_dev=0
for arg ; do
    case "$arg" in
        --spack-dev)
                    spack_dev=1
        ;;
        --spack-reinstall)
                    spack_reinstall=1
        ;;
        --coek-dev)
                    coek_dev=1
        ;;
        --help)
                    echo "build_dev.sh [--spack-reinstall] [--spack-dev] [--coek-dev] [--help]"
                    exit 
        ;;
        *)
                    echo "unknown option: ${arg}"
                    exit 
        ;;
    esac
done

export SPACK_HOME=`pwd`/spack
echo "SPACK_HOME=${SPACK_HOME}"
if [[ -z "${GUROBI_HOME}" ]]; then
    with_gurobi=""
else
    with_gurobi="+gurobi"
fi

if [[ "$spack_reinstall" -eq 1 ]]; then
    rm -Rf spack
fi
if test -d ${SPACK_HOME}; then
    echo ""
    echo "WARNING: Spack directory exists."
    echo ""
else
    echo ""
    echo "Installing Chmmpp dependencies using Spack"
    echo ""
    if [[ "$spack_dev" -eq 0 ]]; then
        git clone https://github.com/or-fusion/spack.git
    else
        git clone git@github.com:or-fusion/spack.git
    fi
    . ${SPACK_HOME}/share/spack/setup-env.sh
    spack env create dev
    spack env activate dev
    if [[ "$coek_dev" -eq 0 ]]; then
        spack add coek@dev ${with_gurobi}
    else
        spack add asl cppad fmt rapidjson highs
    fi
    spack add catch2
    spack install
    spack env deactivate
fi

if [[ "$coek_dev" -eq 1 ]]; then
    \rm -Rf coek
    git clone git@github.com:sandialabs/coek.git
    cd coek
    git checkout dev-weh
    git pull
    mkdir build
    cd build
    if [[ -z "${GUROBI_HOME}" ]]; then
        with_gurobi="OFF"
    else
        with_gurobi="ON"
    fi
    cmake -DCMAKE_PREFIX_PATH=${SPACK_HOME}/var/spack/environments/dev/.spack-env/view -Dwith_highs=ON -Dwith_cppad=OFF -Dwith_fmtlib=ON -Dwith_rapidjson=ON -Dwith_catch2=ON -Dwith_tests=ON -Dwith_asl=ON -Dwith_openmp=OFF -Dwith_gurobi=${with_gurobi} ..
    make -j20
    make install
fi
export COEK_INSTALL="`pwd`/coek/build/install;"

echo ""
echo "Building Chmmpp"
echo ""
\rm -Rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=${COEK_INSTALL}${SPACK_HOME}/var/spack/environments/dev/.spack-env/view -Dwith_coek=ON -Dwith_boost=ON -Dwith_tests=ON ..
make -j20
