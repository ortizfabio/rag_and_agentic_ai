"""This is the installation toolset for this project."""
from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name='develop_generative_ai_applications_get_started',
      version='0.1.0',
      author='db2admin',
      description='Source https://www.coursera.org/learn/develop-generative-ai-applications-get-started/home/module/2',
      long_description=long_description,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Add your dependencies here or leave empty
        # They should match requirements.txt
    ],
    extras_require={
        "dev": [
            "python-dotenv",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pylint>=2.17.0",
            "isort>=5.12.0",
            "rst-to-myst[sphinx]",
        ],
    },
)
