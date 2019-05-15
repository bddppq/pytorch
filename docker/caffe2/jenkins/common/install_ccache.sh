#!/bin/bash

set -ex

# Install sccache from pre-compiled binary.
curl https://s3.amazonaws.com/ossci-linux/sccache -o /usr/local/bin/sccache
chmod a+x /usr/local/bin/sccache

# Setup SCCACHE
###############################################################################
SCCACHE="$(which sccache)"
if [ -z "${SCCACHE}" ]; then
  echo "Unable to find sccache..."
  exit 1
fi

# List of compilers to use sccache on.
compilers=("cc" "c++" "gcc" "g++" "x86_64-linux-gnu-gcc")

# If cuda build, add nvcc to sccache.
if [[ "${BUILD_ENVIRONMENT}" == *-cuda* ]]; then
  compilers+=("nvcc")
fi

if [[ "${BUILD_ENVIRONMENT}" == *-rocm* ]]; then
  compilers+=("hcc")
fi

for compiler in "${compilers[@]}"; do
  if ! hash "${compiler}" 2>/dev/null; then
    continue
  fi
  CACHE_WRAPPER_PATH="$(readlink -e $(which $compiler))"
  if grep "${SCCACHE}" "${CACHE_WRAPPER_PATH}"; then
    echo >&2 "Skip ${CACHE_WRAPPER_PATH}, since it's already an sccache wrapper"
    continue
  fi
  REAL_BINARY_PATH="${CACHE_WRAPPER_PATH}.orig"
  mv "$CACHE_WRAPPER_PATH" "$REAL_BINARY_PATH"

  # Create sccache wrapper.
  (
    echo "#!/bin/sh"
    echo "exec $SCCACHE $REAL_BINARY_PATH \"\$@\""
  ) > "$CACHE_WRAPPER_PATH"
  chmod +x "$CACHE_WRAPPER_PATH"
done
