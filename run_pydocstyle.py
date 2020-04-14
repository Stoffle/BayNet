import subprocess

import pydocstyle

subprocess.call(["pydocstyle", "baynet", "--match=.*(?<!_pb2)\.py"])
