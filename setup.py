from distutils.core import setup

setup(
    name='c3po',
    version='0.1dev',
    packages=['c3po' ],
    long_description=open('README.md').read(),
    install_requires=[
        'cma',
        'cython',
        'matplotlib',
        'numpy',
        'qutip',
        'scipy',
        'tensorflow',
        'tensorflow_probability',
        'uuid'
    ]
)
