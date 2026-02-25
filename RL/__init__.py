# RL package – register lowercase alias so internal cross-imports work
import sys
import importlib

_this = sys.modules[__name__]
# Allow both `import RL.x` and `import rl.x` to resolve to the same package
if 'rl' not in sys.modules:
    sys.modules['rl'] = _this
