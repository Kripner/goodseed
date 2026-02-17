"""Command-line interface for Goodseed.

Usage:
    goodseed [dir]               - Start the local server (alias for 'goodseed serve')
    goodseed serve [dir]         - Start the local server
    goodseed list [dir]          - List projects (or runs with --project)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from goodseed.config import get_projects_dir
from goodseed.server import _scan_projects, _scan_runs, run_server


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the local HTTP server."""
    projects_dir = Path(args.dir) if args.dir else get_projects_dir()
    run_server(projects_dir, port=args.port, verbose=args.verbose)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List projects, or runs within a specific project."""
    projects_dir = Path(args.dir) if args.dir else get_projects_dir()

    if not projects_dir.exists():
        print(f"Projects directory does not exist: {projects_dir}")
        return 0

    if args.project:
        # List runs for a specific project
        runs = _scan_runs(projects_dir)
        runs = [r for r in runs if r["project"] == args.project]

        if not runs:
            print(f"No runs found in project '{args.project}'.")
            return 0

        for run in runs:
            status = run.get("status", "unknown")
            run_id = run.get("run_id", "?")
            experiment_name = run.get("experiment_name")
            created_at = run.get("created_at") or "-"

            print(f"  [{status}] {run_id}")
            if experiment_name:
                print(f"      name: {experiment_name}")
            print(f"      created: {created_at[:19] if len(created_at) > 19 else created_at}")

        print(f"\n{len(runs)} run(s) in {args.project}")
    else:
        # List projects
        projects = _scan_projects(projects_dir)

        if not projects:
            print("No projects found.")
            return 0

        for proj in projects:
            count = proj["run_count"]
            modified = proj.get("last_modified") or "-"
            if len(modified) > 19:
                modified = modified[:19]
            print(f"  {proj['name']}  ({count} run{'s' if count != 1 else ''}, last modified: {modified})")

        print(f"\n{len(projects)} project(s)")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="goodseed",
        description="Goodseed ML experiment tracker",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the local server")
    serve_parser.add_argument(
        "dir", nargs="?",
        help="Directory containing run databases (default: ~/.goodseed/projects)",
    )
    serve_parser.add_argument(
        "--port", type=int, default=8765,
        help="Port to listen on (default: 8765)",
    )
    serve_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print extra startup information",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List projects (or runs with --project)")
    list_parser.add_argument(
        "dir", nargs="?",
        help="Directory containing run databases (default: ~/.goodseed/projects)",
    )
    list_parser.add_argument(
        "-p", "--project",
        help="List runs within a specific project (e.g. workspace/project-name)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Default: serve
        args.dir = None
        args.port = 8765
        args.verbose = False
        return cmd_serve(args)

    if args.command == "serve":
        return cmd_serve(args)
    elif args.command == "list":
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
