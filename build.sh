#!/bin/bash -e
#
# This scripts builds chmmpp in the `_build` directory to support local
# development and debugging.
#
# This uses Spack to install third-party dependencies in the `_spack` directory.
#
coek_dev=0
debug="OFF"
spack_dev=0
spack_home=`pwd`/_spack
spack_reinstall=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
                    echo "build_dev.sh [--help] [--coek-dev] [--debug] [--spack-dev] [--spack-home <dir>] [--spack-reinstall]"
                    exit 
        ;;
        --coek-dev)
                    coek_dev=1
                    shift
        ;;
        --debug)
                    debug="ON"
                    shift
        ;;
        --spack-dev)
                    spack_dev=1
                    shift
        ;;
        --spack-home)
                    spack_home="$2"
                    shift
                    shift
        ;;
        --spack-reinstall)
                    spack_reinstall=1
                    shift
        ;;
        *)
                    echo "unknown option: ${arg}"
                    exit 
        ;;
    esac
done

#
# Setup directories
#
export SPACK_HOME=${spack_home}
echo "SPACK_HOME=${SPACK_HOME}"
if [[ "${spack_reinstall}" -eq 1 ]]; then
    \rm -Rf ${SPACK_HOME}
fi
\rm -Rf _build
#
# Configure gurobi
#
if [[ -z "${GUROBI_HOME}" ]]; then
    with_gurobi=""
else
    with_gurobi="+gurobi"
fi
#
# Install spack
#
if test -d ${SPACK_HOME}; then
    echo ""
    echo "WARNING: Spack directory exists."
    echo ""
else
    echo ""
    echo "Installing Chmmpp dependencies using Spack"
    echo ""
    #
    # Get Spack
    #
    git clone https://github.com/spack/spack.git ${SPACK_HOME}
    . ${SPACK_HOME}/share/spack/setup-env.sh
    #
    # Get Spack TPLs
    #
    if [[ "$spack_dev" -eq 0 ]]; then
        \rm -Rf _spack_tpls
        git clone https://github.com/or-fusion/or-fusion-spack-repo.git _spack_tpls
    else
        git clone git@github.com:or-fusion/or-fusion-spack-repo.git _spack_tpls
        cd _spack_tpls
        git checkout dev
        git pull
        cd ..
    fi
    spack repo add _spack_tpls/repo || true
    spack repo list
    #
    # Create environment
    #
    spack env create dev
    spack env activate dev
    if [[ "$coek_dev" -eq 0 ]]; then
        spack add coek@dev ${with_gurobi}
    fi
    spack add catch2
    spack install
    spack env deactivate
fi
prefix_path=${SPACK_HOME}/var/spack/environments/dev/.spack-env/view

if [[ "$coek_dev" -eq 1 ]]; then
    #
    # Update spack
    #
    . ${SPACK_HOME}/share/spack/setup-env.sh
    spack repo add _spack_tpls/repo || true
    spack env activate dev
    spack add asl cppad fmt rapidjson highs
    spack install
    spack env deactivate
    spack repo remove _spack_tpls/repo || true
    #
    # Install coek
    #
    if test -d _coek; then
        echo ""
        echo "Using existing _coek clone"
        echo ""
    else
        echo ""
        echo "Cloning into _coek"
        echo ""
        git clone git@github.com:sandialabs/coek.git _coek
    fi
    cd _coek
    git checkout dev-weh
    git pull
    ./build.sh --spack-home ${SPACK_HOME} --spack-env dev
    prefix_path="`pwd`/_build/install;${prefix_path}"
    cd ..
fi

echo ""
echo "Building Chmmpp"
echo ""
mkdir _build
cd _build
cmake -DCMAKE_PREFIX_PATH=${prefix_path} -Dwith_coek=ON -Dwith_boost=ON -Dwith_tests=ON -Dwith_debug=${debug} ..
make -j20
