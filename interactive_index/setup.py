from setuptools import setup, find_packages

setup(
    name="interactive_index",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "faiss-gpu",
        "numpy==1.18.5",
        "PyYAML>=5.3.1",
    ],  # TODO: put reqs here
    python_requires=">=3.7",
)
