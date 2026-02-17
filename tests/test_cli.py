"""Unit tests for CLI commands."""

import pytest

from goodseed.cli import main
from goodseed.run import Run


class TestCmdList:
    def test_list_empty(self, tmp_path, capsys):
        main(["list", str(tmp_path)])
        output = capsys.readouterr().out
        assert "No projects found" in output

    def test_list_projects(self, tmp_path, capsys):
        with Run(run_name="cli-run", goodseed_home=tmp_path):
            pass

        projects_dir = tmp_path / "projects"
        main(["list", str(projects_dir)])
        output = capsys.readouterr().out
        assert "default" in output
        assert "1 project(s)" in output

    def test_list_runs_with_project(self, tmp_path, capsys):
        with Run(
            experiment_name="cli-exp",
            run_name="cli-run",
            goodseed_home=tmp_path,
        ) as r:
            r.log_configs({"lr": 0.001})

        projects_dir = tmp_path / "projects"
        main(["list", str(projects_dir), "--project", "default"])
        output = capsys.readouterr().out
        assert "cli-run" in output
        assert "cli-exp" in output
        assert "1 run(s)" in output

    def test_list_multiple_runs(self, tmp_path, capsys):
        with Run(run_name="run-a", goodseed_home=tmp_path):
            pass
        with Run(run_name="run-b", goodseed_home=tmp_path):
            pass

        projects_dir = tmp_path / "projects"
        main(["list", str(projects_dir), "-p", "default"])
        output = capsys.readouterr().out
        assert "run-a" in output
        assert "run-b" in output
        assert "2 run(s)" in output

    def test_list_nonexistent_dir(self, tmp_path, capsys):
        main(["list", str(tmp_path / "nonexistent")])
        output = capsys.readouterr().out
        assert "does not exist" in output


class TestCreateParser:
    def test_default_command(self):
        """When no command given, args.command should be None (defaults to serve)."""
        from goodseed.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_serve_with_port(self):
        from goodseed.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["serve", "--port", "9999"])
        assert args.command == "serve"
        assert args.port == 9999

    def test_list_with_dir(self):
        from goodseed.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(["list", "/some/path"])
        assert args.command == "list"
        assert args.dir == "/some/path"
