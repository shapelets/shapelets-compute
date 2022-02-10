import sys

import auditwheel
from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

# libs loaded dynamically; do not include it
for p in POLICIES:
   p['lib_whitelist'].append('libcufft.so.10')  # shapelets_compute.libs/libcufft-da2333af.so.10.5.0.43

if __name__ == "__main__":
    sys.exit(main())
