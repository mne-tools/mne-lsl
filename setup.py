from setuptools import find_packages, setup

setup(
    name='NeuroDecode',
    version='0.9dev',
    author='Kyuhwa Lee, Arnaud Desvachez',
    author_email='lee.kyuh@gmail.com, arnaud.desvachez@gmail.com',
    license='The GNU General Public License',
    url='https://github.com/fcbg-hnp/NeuroDecode/',
    description='Real-time brain-machine interface',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        'h5py>=2.7',
        'opencv_python>=3.4',
        'numpy>=1.16',
        'scipy>=1.2',
        'colorama>=0.3.9',
        'xgboost>=0.81',
        'matplotlib>=3.0.2',
        'mne>=0.16',
        'psutil>=5.4.8',
        'setuptools>=39.0.1',
        'pyqtgraph>=0.10.0',
        'pylsl>=1.12.2',
        'pywin32>=222 ; platform_system=="Windows"',
        'ipython>=6',
        'PyQt5>=5',
        'pyxdf>=1.15.2',
        'pyserial>=3.4',
        'simplejson>=3.16.0',
        'scikit_learn>=0.21',
        'future',
        'configparser'
    ]
)
