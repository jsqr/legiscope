#!/usr/bin/env python3
"""
Command-line script to create jurisdiction directory structures.
"""

import argparse
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legiscope.utils import create_jurisdiction_structure


def main():
    """Create jurisdiction directory structure from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create directory structure for a new jurisdiction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s NY "New York"
  %(prog)s CA LosAngeles
  %(prog)s IL Chicago --verbose
        """,
    )

    parser.add_argument(
        "state", help="Two-letter state abbreviation (e.g., NY, CA, IL)"
    )
    parser.add_argument(
        "municipality", help="Municipality name (e.g., New York, LosAngeles)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        base_path = create_jurisdiction_structure(args.state, args.municipality)
        print(f"Successfully created jurisdiction structure: {base_path}")

        if args.verbose:
            subdirs = ["raw", "processed", "tables"]
            for subdir in subdirs:
                subdir_path = Path(base_path) / subdir
                print(f"   📁 {subdir_path}")

    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
