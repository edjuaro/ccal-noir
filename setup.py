from setuptools import setup

version = {}
with open("ccalnoir/version.py") as fp:
    exec(fp.read(), version)
# later on we use: version['__version__']
__version__ = version['__version__']

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
