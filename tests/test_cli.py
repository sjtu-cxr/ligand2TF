import subprocess
import sys


def test_cli_help_lists_workflows():
    result=subprocess.run([sys.executable,'-m','ligand2tf','--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    for command in ['predict','evaluate','train-gate','verify','train-dstar','score-dstar','example','bundle','prepare-protein']:
        assert command in result.stdout
