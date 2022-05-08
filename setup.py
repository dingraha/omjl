import os
import setuptools
import pyjuliapkg

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="omjl",
    version="0.1.0",
    author="Daniel Ingraham and contributors",
    author_email="d.j.ingraham@gmail.com",
    description="Create OpenMDAO Components in the Julia programming language",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/dingraha/omjl.git",
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "openmdao",
    ],
)

# Install Julia dependencies.
pyjuliapkg.resolve()
