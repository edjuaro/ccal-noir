from setuptools import setup

execfile('...sample/version.py')
# now we have a `__version__` variable
# later on we use: __version__

setup(name='ccalnoir',
      version=__version__,
      description='CCAL but with "No R".',
      url='https://github.com/edjuaro/ccal-noir',
      author='Edwin F. Juarez',
      author_email='ejuarez@ucsd.edu',
      license='Modified BSD',
      packages=['ccalnoir'],
      zip_safe=False,
      install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'matplotlib',
            'statsmodels',
            'seaborn',
            'validators',
            'ipython',
            'genepattern-notebook',
            'genepattern-python',
            ],
      )
