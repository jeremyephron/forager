import setuptools

setuptools.setup(
    name='forager',
    version='0.0.0',
    install_requires=[
        'faiss-gpu>=1.6.0',
        'tensorflow-gpu>=2.3.1',
        'numpy==1.18.5',
        'PyYAML>=5.3.1'
    ],  # TODO: put reqs here
    packages=['interactive_index'],
)

