# Standard Library
import os.path as osp
from pathlib import Path

# Third Party Library
from invoke import task


@task
def sanity(cx) -> None:
    """Perform sanity check for jubeo"""

    print("All systems go!")


@task
def pin_tool_deps(cx) -> None:
    """Pins or upgrades the requirements.txt for the jubeo tooling from
    the requirements.in (from the upstream repo) and the
    local.requirements.in (for project specific tooling dependencies)
    files."""

    req_in = Path('.jubeo') / "requirements.in"
    local_req_in = Path('.jubeo') / "local.requirements.in"
    req_txt = Path('.jubeo') / "requirements.txt"

    assert osp.exists(req_in), "No 'requirements.in' file"

    # add the local reqs if given
    if osp.exists(local_req_in):
        req_str = f"{req_in} {local_req_in}"
    else:
        req_str = req_in

    cx.run(f"pip-compile --upgrade --output-file={req_txt} {req_str}")
