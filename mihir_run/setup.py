from setuptools import setup, find_packages

setup(
    name="knn",
    version="0.1",
    description="Fast embedding queries.",
    url="https://github.com/gtmtg/knn",
    author="Mihir Garimella",
    author_email="mihirg@stanford.edu",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["aiohttp[speedups]", "dataclasses-json", "numpy", "runstats"],
    python_requires=">=3.7",
    zip_safe=True,
)
