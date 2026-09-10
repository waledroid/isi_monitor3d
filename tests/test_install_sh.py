"""Runs the hermetic shell checks for install.sh under pytest."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_install_sh_shell_suite():
    r = subprocess.run(["bash", str(ROOT / "tests/shell/test_install_sh.sh")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 failed" in r.stdout
