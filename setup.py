from distutils.core import setup

setup(
    name='c3po',
    version='1.0rc',
    packages=[
        'c3po',
        'c3po/generator',
        'c3po/libraries',
        'c3po/optimizers',
        'c3po/schemas',
        'c3po/signal',
        'c3po/system',
        'c3po/utils'
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
