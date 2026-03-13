from distutils.core import setup

setup(
    name='qubayes',
    version='0.1',
    packages=['qubayes'],
    package_data={
        'qubayes': ['*.txt', '*.json', '*.csv']  # include if any non-code files exist
    },
    url='',
    license='GPL',
    author='krf',
    author_email='florian.krebs@joanneum.at',
    description='Quantum Bayesian Network Tools'
)