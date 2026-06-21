"""Run and visualize a trained dreamer.py actor on ball-balance.

This is a convenience entry point around ``scripts/visualize_dreamer_policy.py``.
It mirrors the usual Dreamer evaluation command, but saves an online dashboard GIF
and metrics by default.
"""

from scripts.visualize_dreamer_policy import main


if __name__ == "__main__":
    main()
