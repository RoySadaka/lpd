import os
from setuptools import setup, find_packages
from io import open as io_open

src_dir = os.path.abspath(os.path.dirname(__file__))

install_requires = []

path_requirements_dev = os.path.join(src_dir, 'requirements-dev.txt')
path_requirements_dev_no_deps = os.path.join(src_dir, 'requirements-dev-nodeps.txt')
for path in [path_requirements_dev, path_requirements_dev_no_deps]:
    with io_open(path, mode='r') as fd:
        for i in fd.read().strip().split('\n'):
            req = i.strip().split('#', 1)[0].strip()
            install_requires.append(req)


README_md = ''
fndoc = os.path.join(src_dir, 'README.md')
with io_open(fndoc, mode='r', encoding='utf-8') as fd:
    README_md = fd.read()


classifiers=[
        # (https://pypi.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities'
    ]

setup(
    name='lpd',
    version='0.4.2',
    description='A Fast, Flexible Trainer with Callbacks and Extensions for PyTorch',
    long_description_content_type='text/markdown',
    long_description=README_md,
    license='MIT Licences',
    url='https://github.com/roysadaka/lpd',
    author='Roy Sadaka',
    maintainer='lpd developers',
    maintainer_email='torch.lpd@gmail.com',
    packages=find_packages(exclude=['tests', 'tests/*', 'examples', 'examples/*']),
    install_requires=install_requires,
    python_requires='>=3.6',
    classifiers=classifiers,
    keywords=['pytorch,trainer,callback,callbacks,earlystopping,tensorboard,modelcheckpoint,checkpoint,layers,dense,metrics,predictor,binary accuracy,extensions,track,monitor,machine,deep learning,neural,networks,AI,keras decay,confusion matrix']
)