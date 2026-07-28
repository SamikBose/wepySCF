# Standard Library
import os
from pathlib import Path

# Third Party Library
from invoke import task


@task
def init(cx) -> None:
    cx.run("mkdir -p _output")
    cx.run("mkdir -p _tangle_source")

@task
def clean(cx) -> None:
    cx.run("rm -rf _output/*")
    cx.run("rm -rf _tangle_source/*")

@task(pre=[init])
def tangle(cx) -> None:
    cx.run("jupyter-nbconvert --to 'python' --output-dir=_tangle_source README.ipynb")


@task
def clean_env(cx) -> None:
    cx.run("rm -rf _env")

@task(pre=[init])
def env(cx) -> None:
    """Create the environment from the specs in 'env'. Must have the
    entire repository available as it uses the tooling from it.

    """

    example_name = Path(os.getcwd()).stem

    with cx.cd("../../../"):
        cx.run(f"inv docs.env-tutorial -n {example_name}")
