import subprocess
import sys


def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "nqs_vmc.cli", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "run" in r.stdout and "plot" in r.stdout


def test_cli_missing_config_errors():
    r = subprocess.run([sys.executable, "-m", "nqs_vmc.cli", "run",
                        "--config", "does_not_exist.yaml"],
                       capture_output=True, text=True)
    assert r.returncode != 0