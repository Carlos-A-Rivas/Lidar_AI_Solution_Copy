#!/bin/bash
# Mirror of tool/run.sh but using the custom environment and executable
. tool/custom_environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

set -e

mkdir -p build
cd build
cmake ..
make -j
cd ..

./build/custom_bevfusion $DEBUG_DATA $DEBUG_MODEL $DEBUG_PRECISION
