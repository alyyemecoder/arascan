from setuptools import setup, find_packages

setup(
    name="arascan",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'python-multipart>=0.0.5',
        'python-jose[cryptography]>=3.3.0',
        'passlib[bcrypt]>=1.7.4',
    ],
)
