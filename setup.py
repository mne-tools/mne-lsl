from setuptools import find_packages, setup

setup(
    name='pycnbi',
    version='0.6dev',
    author='Kyuhwa Lee',
    author_email='lee.kyuh@gmail.com',
    license='The GNU General Public License',
    url='https://github.com/dbdq/pycnbi/',
    description='Real-time brain-machine interface',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=['mne>=0.14', 'scipy', 'numpy', 'pylsl',\
        'opencv-python>=3.3', 'future', 'pyqtgraph', 'matplotlib>=2.1.0',\
        'configparser', 'psutil'],
)
