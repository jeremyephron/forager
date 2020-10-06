from setuptools import setup, find_packages

setup(
    name="interactive_index",
    version="0.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "faiss-gpu>=1.6.0",
        "tensorflow-gpu>=2.3.0",
        "numpy==1.18.5",
        "PyYAML>=5.3.1",
    ],  # TODO: put reqs here
)
