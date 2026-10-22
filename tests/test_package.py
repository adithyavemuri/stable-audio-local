"""Regression tests for portable orchestration and source protection."""
import contextlib
import importlib.util
import io
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from stable_audio_local import cli, runtime

PACKAGE = Path(runtime.__file__).parent


def load_helpers():
    spec = importlib.util.spec_from_file_location("test_worker_helpers", PACKAGE / "_worker_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OrchestrationTests(unittest.TestCase):
    def test_setup_dry_run_never_executes_or_creates_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "workspace with spaces"
            output = io.StringIO()
            with patch("subprocess.run") as execute, contextlib.redirect_stdout(output):
                code = cli.main(["--workspace", str(target), "--uv", "uv", "--dry-run", "setup"])
            self.assertEqual(code, 0)
            execute.assert_not_called()
            self.assertFalse(target.exists())
            commands = [json.loads(line)["command"] for line in output.getvalue().splitlines()]
            self.assertTrue(any(runtime.SPEC["commit"] in command for command in commands))
            self.assertTrue(any(str(target / runtime.SPEC["folder"]) in command for command in commands))
            joined = " ".join(str(item) for command in commands for item in command)
            self.assertNotIn("--all-extras", joined)
            self.assertNotIn("--extra lora", joined)
            if runtime.SPEC["kind"] == "stable":
                self.assertIn("cp311-cp311-win_amd64.whl", joined)
                self.assertNotIn("linux_x86_64.whl", joined)
            else:
                self.assertIn("--no-install-package peft", joined)

    def test_worker_accepts_wav_path_after_workspace_option(self):
        with patch.object(sys, "path", [str(PACKAGE), *sys.path]):
            spec = importlib.util.spec_from_file_location("worker_parse_test", PACKAGE / "_worker.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parsed = module.arguments(["validate", "--workspace", "workspace with spaces", "recording with spaces.wav"])
            self.assertEqual(parsed.paths, ["recording with spaces.wav"])

    def test_invalid_count_stops_before_any_worker(self):
        with patch.object(cli, "worker") as worker, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["benchmark", "--count", "0"])
            worker.assert_not_called()

    def test_workspace_paths_remain_single_arguments(self):
        args = SimpleNamespace(workspace=Path("folder with spaces"), dry_run=False)
        with patch.object(runtime, "python_path", return_value=Path(__file__)), patch.object(runtime, "run") as execute:
            runtime.worker(args, "doctor")
        command = execute.call_args.args[0]
        self.assertIn(args.workspace, command)

    def test_credentials_are_not_put_in_workspace_or_command(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(runtime.Path, "home", return_value=Path("user-home")):
            env = runtime.environment(Path("new-workspace"))
        self.assertNotIn("new-workspace", env["HF_TOKEN_PATH"])
        self.assertNotIn("HF_TOKEN", env)

    def test_wrong_remote_is_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            (workspace / runtime.SPEC["folder"]).mkdir()
            args = SimpleNamespace(workspace=workspace, dry_run=False)
            with patch.object(runtime.sys, "platform", "win32"), patch.object(runtime.subprocess, "check_output", return_value="https://example.com/unofficial.git\n"), patch.object(runtime, "run") as mutate:
                with self.assertRaisesRegex(RuntimeError, "official"):
                    runtime.setup_repo(args)
                mutate.assert_not_called()

    def test_existing_dirty_repo_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            (workspace / runtime.SPEC["folder"]).mkdir()
            args = SimpleNamespace(workspace=workspace, dry_run=False)
            with patch.object(runtime.sys, "platform", "win32"), patch.object(runtime.subprocess, "check_output", side_effect=[runtime.SPEC["upstream"], " M source.py\n"]), patch.object(runtime, "run") as mutate:
                with self.assertRaisesRegex(RuntimeError, "tracked changes"):
                    runtime.setup_repo(args)
                mutate.assert_not_called()

    def test_source_cannot_be_output_even_with_overwrite(self):
        helpers = load_helpers()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            source.write_bytes(b"original recording")
            args = SimpleNamespace(output_dir=Path(folder), workspace=Path(folder), source=source, overwrite=True)
            with self.assertRaisesRegex(ValueError, "source"):
                helpers.output_path(args, "source.wav")
            self.assertEqual(source.read_bytes(), b"original recording")

    def test_changed_source_is_detected(self):
        helpers = load_helpers()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            source.write_bytes(b"original")
            initial = helpers.digest(source)
            helpers.source_unchanged(source, initial)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                helpers.source_unchanged(source, initial)

    def test_existing_output_requires_explicit_overwrite(self):
        helpers = load_helpers()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "take.wav"
            target.write_bytes(b"previous take")
            args = SimpleNamespace(output_dir=Path(folder), workspace=Path(folder), source=None, overwrite=False)
            with self.assertRaises(FileExistsError):
                helpers.output_path(args, target.name)
            self.assertEqual(target.read_bytes(), b"previous take")

    def test_unexpected_upstream_version_is_rejected(self):
        helpers = load_helpers()
        with patch.object(helpers.subprocess, "check_output", return_value="wrong-commit\n"):
            with self.assertRaisesRegex(RuntimeError, "Untested"):
                helpers.repo_commit(Path("repo"), "expected")


if __name__ == "__main__":
    unittest.main()
