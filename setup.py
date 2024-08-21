from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'doubleTime'
LONG_DESCRIPTION = 'doubleTime is a method to estimate the timing of whole-genome doubling event(s) on a clone tree using SNVs.'

setup(
    name="doubleTime", 
    version=VERSION,
    author="Andrew McPherson",
    author_email="<mcphera1@mskcc.org>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
)
