#!/bin/bash

set -euo pipefail

set -x
pushd $(dirname $0)

podman build --ulimit nofile=$(ulimit -n):$(ulimit -n) -t lukasgd/ngc-sc22-dl-tutorial:24.05 -f Dockerfile .
enroot import -x mount -o /iopsstor/scratch/cscs/lukasd/images/lukasgd+ngc-sc22-dl-tutorial+24.05.sqsh podman://lukasgd/ngc-sc22-dl-tutorial:24.05

popd
set +x
