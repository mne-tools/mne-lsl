#!/bin/bash

set -eo pipefail

if [ "${MNE_LSL_LIBLSL_BUILD_UNITTESTS}" != "1" ]; then
    echo "Skipping liblsl unit tests."
    exit 0
fi

echo "Running liblsl unit tests..."
SCRIPT_DIR="$(dirname "$0")"
TEST_DIR="${SCRIPT_DIR}/../tests/liblsl"
TEST_ARGS="--order rand --wait-for-keypress never --durations yes"

# set LD_LIBRARY_PATH or DYLD_LIBRARY_PATH
if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH="${TEST_DIR}:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${TEST_DIR}:${LD_LIBRARY_PATH}"
fi

# run the test binaries
for TEST_FILE_BASENAME in "lsl_test_internal" "lsl_test_exported"; do
    if [ -f "${TEST_DIR}/${TEST_FILE_BASENAME}" ]; then
        TEST_FILE="${TEST_DIR}/${TEST_FILE_BASENAME}"
    elif [ -f "${TEST_DIR}/${TEST_FILE_BASENAME}.exe" ]; then
        TEST_FILE="${TEST_DIR}/${TEST_FILE_BASENAME}.exe"
    else
        echo "Error: Test file ${TEST_FILE_BASENAME} not found in ${TEST_DIR}."
        exit 1
    fi

    echo "Running ${TEST_FILE}..."
    "${TEST_FILE}" ${TEST_ARGS}
    TEST_EXIT_CODE=$?
    if [ ${TEST_EXIT_CODE} -ne 0 ]; then
        echo "Test ${TEST_FILE} failed with exit code ${TEST_EXIT_CODE}."
        exit ${TEST_EXIT_CODE}
    fi
done

echo "All liblsl unit tests passed."
exit 0
