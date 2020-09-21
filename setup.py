from distutils.core import setup

setup(
    name='c3po',
    version='1.0rc',
    packages=[
        'c3po'
    ],
    long_description=open('README.md').read(),
    install_requires=[
        'tensorflow',
        'tensorflow-probability',
        'cma',
        'cython',
        'matplotlib',
        'numpy',
        'scipy',
        'adaptive'
    ]
)
