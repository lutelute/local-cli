"""Entry point for python -m local_cli."""

from local_cli import __version__


def main() -> None:
    """Run the local-cli application."""
    print(f"local-cli v{__version__}")


if __name__ == "__main__":
    main()
