from setuptools import find_packages, setup
import os

setup(
    name='pycnbi',
    version='0.5dev',
    author='Kyuhwa Lee',
    author_email='lee.kyuh@gmail.com',
    license='The GNU General Public License',
    url='https://c4science.ch/source/pycnbi/',
    description='Real-time brain-machine interface',
    long_description=open('README.txt').read(),
    packages=find_packages(),
    install_requires=['mne>=0.14', 'scipy', 'numpy', 'opencv-python>=3.3',\
        'future', 'pyqtgraph', 'matplotlib>=2.1.0'],
)
