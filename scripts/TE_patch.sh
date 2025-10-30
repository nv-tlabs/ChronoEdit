#!/bin/bash

# 1) Find your conda nvrtc
CONDA_NVRTC=$(python - <<'PY'
import glob, sys, os
cand = glob.glob(os.path.join(sys.prefix, "lib", "libnvrtc.so*"))
print(cand[0] if cand else "")
PY)
echo "Found: $CONDA_NVRTC"

# 2) Make a shim dir early in PATH
mkdir -p $HOME/bin
cat > $HOME/bin/ldconfig <<'SH'
#!/usr/bin/env bash
if [[ "$*" == *"-p"* ]]; then
  # Print a minimal entry so "grep 'libnvrtc'" succeeds.
  # Adjust the path below to your env's libnvrtc
  echo "libnvrtc.so.12 (libc6,x86-64) => ${CONDA_NVRTC}"
  exit 0
fi
# Fallback to real ldconfig if needed
exec /sbin/ldconfig "$@"
SH
chmod +x $HOME/bin/ldconfig

# 3) Make the var visible inside the shim at runtime
#    (bake it into the file)
sed -i.bak "s|\${CONDA_NVRTC}|$CONDA_NVRTC|g" $HOME/bin/ldconfig

# 4) Prepend to PATH for this session
export PATH="$HOME/bin:$PATH"

# 5) Try the import
python -c "import transformer_engine as te; print('TE import OK')"