#!/bin/bash

set -eo pipefail

if [[ "$1" == "" ]] || [[ "$2" == "" ]]; then
    echo "Usage: $0 <ARCH> <SHA>"
    exit 1
fi

ARCH=$1
SHA=$2
MSG="$(git show -s --format=%s $SHA)"
# If it's a PR, and aarch64 architecture, and no [aarch64] in the commit message, just build and test one aarch64 wheel with one tox config
if [[ "$ARCH" == "aarch64" ]] && [[ "$GITHUB_EVENT_NAME" == "pull_request" ]] && [[ "$MSG" != *'[aarch64]'* ]]; then
    echo "Skipping $ARCH build for $GITHUB_EVENT_NAME for speed ($MSG)"
    echo "skip=1" >> $GITHUB_OUTPUT
else
    echo "Building ${ARCH} wheel for $GITHUB_EVENT_NAME ($MSG)"
fi
