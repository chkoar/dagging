import shutil

from invoke import task


@task
def dist(ctx):
    ctx.run("python setup.py sdist bdist_wheel")


@task
def publish(ctx):
    ctx.run("pip install 'twine>=1.5.0'")
    ctx.run("twine upload dist/*")
    shutil.rmtree("build")
    shutil.rmtree("dist")
    shutil.rmtree("dagging.egg-info")


@task
def docs(ctx):
    ctx.run("cd docs && make html")


@task
def test(ctx):
    ctx.run("pytest -v --cov --cov-report term-missing")


@task
def lint(ctx):
    ctx.run("black . --check")
    ctx.run("flake8")
