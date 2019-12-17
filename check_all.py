"""Run all checks"""
import subprocess

import pytest
from pylint import lint as pylint
from mypy import api as mypy
import black

MODULE_NAME = "baynet"

# pytest
print("############### pytest #################")
pytest.main(["-vv", f"--cov={MODULE_NAME}", "--cov-fail-under=95", "tests/"])
print("")

# pydocstyle
print("############### pydocstyle #################")
subprocess.call(["pydocstyle", MODULE_NAME])
print("")

# pylint
print("############### pylint #################")
PYLINT_RESULTS = pylint.Run([f"./{MODULE_NAME}/"], do_exit=False)
print("")

# mypy
print("###############  mypy  #################")
MYPY_RESULTS = mypy.run([f"./{MODULE_NAME}/", "--warn-redundant-casts", "--show-error-context"])
print(MYPY_RESULTS[0], end="")
print(MYPY_RESULTS[1], end="")
print("Exit code of mypy: {}".format(MYPY_RESULTS[2]))

# # mypy
# print("############  mypy tests  ##############")
# MYPY_RESULTS = mypy.run(
#     ["./tests/", "--warn-redundant-casts", "--show-error-context", "--check-untyped-defs"]
# )
# print(MYPY_RESULTS[0], end="")
# print(MYPY_RESULTS[1], end="")
# print("Exit code of mypy: {}".format(MYPY_RESULTS[2]))

# black
print("############  black ##############")
black.main(["-l", "100", "-t", "py36", "-S", "./", "--exclude=(.?env|.?venv)"])
print("")
