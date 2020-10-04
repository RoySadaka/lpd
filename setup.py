import os
from setuptools import setup, find_packages
from io import open as io_open

src_dir = os.path.abspath(os.path.dirname(__file__))

install_requires = []
requirements_dev = os.path.join(src_dir, 'requirements-dev.txt')
with io_open(requirements_dev, mode='r') as fd:
    install_requires = [i.strip().split('#', 1)[0].strip() for i in fd.read().strip().split('\n')]

README_md = ''
fndoc = os.path.join(src_dir, 'README.md')
with io_open(fndoc, mode='r', encoding='utf-8') as fd:
    README_md = fd.read()

setup(
    name='lpd',
    version='0.0.9',
    description='A Fast, Flexible Trainer and Extensions for Pytorch',
    long_description_content_type='text/markdown',
    long_description=README_md,
    license='MIT Licences',
    url='https://github.com/roysadaka/lpd',
    author='Roy Sadaka',
    maintainer='lpd developers',
    maintainer_email='torch.lpd@gmail.com',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
    classifiers=[
        # (https://pypi.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Operating System :: Microsoft :: MS-DOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: BSD',
        'Operating System :: POSIX :: BSD :: FreeBSD',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities'
    ],
    keywords='pytorch trainer extensions machine deep learning'
)