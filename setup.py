from setuptools import setup

setup(
    name='Resume Parser',
    version='0.1.0',
    author='Matheus Werner',
    author_email='mwerner@inf.puc-rio.br',
    packages=[
        'resume_parser',
        'resume_parser.segmenter',
        'resume_parser.segmenter.crf',
        'resume_parser.segmenter.bert'
    ],
    scripts=[],
    # url='http://pypi.python.org/pypi/PackageName/',
    # license='LICENSE.txt',
    description='An awesome package that does something',
    # long_description=open('README.txt').read(),
    # install_requires=[
    #     "Django >= 1.1.1",
    #     "pytest",
    # ],
)