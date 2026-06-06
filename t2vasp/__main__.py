"""Allow ``python -m t2vasp`` invocation."""

import sys

from .cli import main

sys.exit(main())
