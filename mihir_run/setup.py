from setuptools import setup, find_packages

setup(
    name="knn",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "aiohttp[speedups]",
        "dataclasses-json",
        "gcloud-aio-auth",
        "kubernetes_asyncio",
        "numpy",
        "runstats",
    ],
    python_requires=">=3.7",
    zip_safe=True,
)
