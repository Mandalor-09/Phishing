from setuptools import setup,find_packages
import os 
hyper_opt_pkg = "git+https://github.com/hyperopt/hyperopt-sklearn"
os.system(f'pip install {hyper_opt_pkg}')

def install_other_pakage():
    with open('requirements.txt','r') as f:
        lines = f.readlines()
        packages = [i.replace('/n','') for i in lines if i != '-e .' and not i.startswith('#')]
        return packages
    
setup(
    name='Phishing Website Detection',
    version='0.1.dev0',
    description="A simple phishing website detection tool",
    author='Mandalor-09',
    author_email='oms421621@gmail.com',
    packages=find_packages(),
    install_requires = install_other_pakage()
)