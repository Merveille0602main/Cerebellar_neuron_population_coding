"""Command-line entry point for BSB network inspection and export."""

from __future__ import annotations

import argparse

from .bsb_legacy import read_published_network
from .export import write_spec
from .reference import verify_published_reference
from .validation import validate_export


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cerebellar-export")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "export"):
        command = commands.add_parser(name)
        command.add_argument("source", help="Path to BSB balanced.hdf5")
        command.add_argument(
            "--verify-published",
            action="store_true",
            help="require the exact published file size and MD5 before reading",
        )
        if name == "export":
            command.add_argument("destination", help="Directory for the portable network specification")
    validate = commands.add_parser("validate")
    validate.add_argument("spec", help="Path to an exported portable network specification")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        manifest = validate_export(args.spec)
        print(
            f"Validated {len(manifest['populations'])} populations and "
            f"{len(manifest['connectivity'])} connectivity sets."
        )
        return 0
    if args.verify_published:
        verify_published_reference(args.source)
        print("Published reference checksum: OK")
    spec = read_published_network(args.source)
    print("Populations:")
    for name, population in sorted(spec.populations.items()):
        print(f"  {name}: {population.size} cells")
    print("Connectivity:")
    for name, connection in sorted(spec.connectivity.items()):
        print(f"  {name}: {connection.size} contacts ({connection.pre_population} -> {connection.post_population})")
    if args.command == "export":
        write_spec(spec, args.destination)
        print(f"Wrote {args.destination}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
