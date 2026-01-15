"""
Entry point for running the package as a module.

Usage:
    python -m household_recovery --help
    python -m household_recovery --households 30 --steps 15
"""

from .cli import main

if __name__ == '__main__':
    main()
