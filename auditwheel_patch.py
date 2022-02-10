# Monkey patch to not ship libjvm.so in pypi wheels
import sys

import auditwheel
from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

# libjvm is loaded dynamically; do not include it
for p in POLICIES:
    p['lib_whitelist'].append('libcufft.so')  # shapelets_compute.libs/libcufft-da2333af.so.10.5.0.43

if __name__ == "__main__":
    sys.exit(main())
