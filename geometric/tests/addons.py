"""
Figures out currently available modules
"""

import pytest
import os


def _plugin_import(plug):
    """
    Tests to see if a module is available
    """
    import sys
    if sys.version_info >= (3, 4):
        from importlib import util
        plug_spec = util.find_spec(plug)
    else:
        import pkgutil
        plug_spec = pkgutil.find_loader(plug)
    if plug_spec is None:
        return False
    else:
        return True


# Modify paths for testing
os.environ["DQM_CONFIG_PATH"] = os.path.dirname(os.path.abspath(__file__))
os.environ["TMPDIR"] = "/tmp/"

# Add flags
using_psi4 = pytest.mark.skipif(
    _plugin_import("psi4") is False, reason="could not find psi4. please install the package to enable tests")
using_rdkit = pytest.mark.skipif(
    _plugin_import("rdkit") is False, reason="could not find rdkit. please install the package to enable tests")
using_qcengine = pytest.mark.skipif(
    _plugin_import("qcengine") is False, reason="could not find qcengine. please install the package to enable tests")

# make tests run in their own folder
def in_folder(func):
    def new_func(*args, **kwargs):
        test_parent_folder = 'test_generated_files'
        if not os.path.exists(test_parent_folder):
            os.mkdir(test_parent_folder)
        os.chdir(test_parent_folder)
        test_name = func.__name__
        if not os.path.exists(test_name):
            os.mkdir(test_name)
        os.chdir(test_name)
        func(*args, **kwargs)
        os.chdir('../..')
    return new_func