from distutils.core import setup

setup(
    name='c3po',
    version='many-birds',
    packages=['c3po'],
    long_description=open('README.md').read(),
    install_requires=[
        'cma',
        'cython',
        'matplotlib',
        'numpy',
        'scipy',
        'tensorflow',
    ]
)
