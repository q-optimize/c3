from distutils.core import setup

setup(
    name='c3po',
    version='0.2dev',
    packages=[
        'c3po',
        'c3po/utils',
        'c3po/system',
        'c3po/libraries',
        'c3po/optimizers',
        'c3po/runs',
        'c3po/signal',
        'c3po/generator'
    ],
    long_description=open('README.md').read(),
    install_requires=[
        'cma',
        'cython',
        'matplotlib',
        'numpy',
        'scipy',
    ]
)
