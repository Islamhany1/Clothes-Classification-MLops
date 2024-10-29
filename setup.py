# When you install a package with -e ., pip looks for a setup.py file in the current folder (.).
# -e Tells pip to install the package by linking directly to your source code rather than copying it. This link allows any changes in the code to reflect immediately without needing a reinstallation.
# This setup.py file makes it easy for others to install and use the code as a package

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Clothes-Classification-Project"
AUTHOR_USER_NAME = "Eslam"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "eslamhany166@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)