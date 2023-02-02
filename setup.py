import setuptools

with open('README.md', 'rt') as fh:
    long_description = fh.read()

with open('catabra/__version__.py', 'rt') as fh:
    version = ''
    for ln in fh.readlines():
        if ln.startswith('__version__ = '):
            version = ln[14:].strip('"\'')

with open('requirements.txt', 'rt') as fr:
    requirements = fr.readlines()

setuptools.setup(
    name="catabra",
    version=version,
    author="Alexander Maletzky",
    license="Apache 2.0 with Commons Clause 1.0",
    author_email="alexander.maletzky@risc-software.at",
    description="Tabular data analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/risc-mi/catabra",
    packages=setuptools.find_packages(exclude=['test', 'examples']),
    install_requires=requirements,
    extra_requires=[],
    platforms=['Linux'],
    classifiers=[
        "License :: Apache 2.0 with Commons Clause 1.0",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.9',
)
