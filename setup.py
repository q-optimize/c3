from distutils.core import setup

setup(
    name="c3",
    version="1.0rc",
    packages=[
        "c3",
        "c3/generator",
        "c3/libraries",
        "c3/optimizers",
        "c3/schemas",
        "c3/signal",
        "c3/system",
        "c3/utils",
    ],
    long_description=open("README.md").read(),
    install_requires=[
        "adaptive==0.11.1",
        "cma==3.0.3",
        "cython",
        "ipython==7.18.1",
        "matplotlib==3.3.2",
        "numpy==1.18.5",
        "scipy==1.5.2",
        "tensorflow==2.3.1",
        "tensorflow-estimator==2.3.0",
        "tensorflow-probability==0.11.1",
    ],
)
