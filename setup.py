from setuptools import setup, find_packages

setup(
    name="fractal_regularization",
    version="0.1.0",
    author="Arjun Shukla",
    author_email="arjunshukla6558@gmail.com",
    description="A fractal regularization for neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arjun988/fractal_regularization",  
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
