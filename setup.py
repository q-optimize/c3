from distutils.core import setup

setup(
    name='c3',
    version='1.0rc',
    packages=[
        'c3',
        'c3/generator',
        'c3/libraries',
        'c3/optimizers',
        'c3/schemas',
        'c3/signal',
        'c3/system',
        'c3/utils'
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
