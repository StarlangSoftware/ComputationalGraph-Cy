from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


setup(
    name='NlpToolkit-ComputationalGraph',
    version='1.0.0',
    packages=['ComputationalGraph'],
    url='https://github.com/StarlangSoftware/ComputationalGraph-Py',
    license='',
    author='olcaytaner',
    author_email='olcay.yildiz@ozyegin.edu.tr',
    description='Computational Graph library',
    long_description=long_description,
    install_requires = ['NlpToolkit-Math'],
    long_description_content_type='text/markdown'
)