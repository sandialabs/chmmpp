#!/bin/bash -e
#
# This scripts builds chmmpp in the `build` directory to support local
# development and debugging.
#
# This uses Spack to install third-party dependencies in the `spack` directory.
#
spack_dev=0
for arg ; do
    case "$arg" in
        --spack-dev)
                    spack_dev=1
        ;;
        --help)
                    echo "build_dev.sh [--python] [--help]"
                    exit 
        ;;
        *)
                    echo "unknown option: ${arg}"
                    exit 
        ;;
    esac
done

if [-n "${GUROBI_HOME}"]; then
    with_gurobi="+gurobi"
else
    with_gurobi=""
fi
   
export SPACK_HOME=`pwd`/spack
echo "SPACK_HOME=${SPACK_HOME}"

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
    spack add coek@dev ${with_gurobi}
    spack add catch2
    spack install
    spack env deactivate
fi

echo "Building Chmmpp"
echo ""
\rm -Rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=${SPACK_HOME}/var/spack/environments/dev/.spack-env/view -Dwith_coek=ON -Dwith_boost=ON ..
make -j20