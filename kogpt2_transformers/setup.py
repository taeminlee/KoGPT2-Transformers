from setuptools import setup, find_packages
from kogpt2_transformers import __version__

with open('requirements.txt') as f:
    require_packages = [line[:-1] if line[-1] == '\n' else line for line in f]

setup(name='kogpt2-transformers',
      version=__version__,
      url='https://github.com/taeminlee/KoGPT2-Transformers',
      license='Apache-2.0',
      author='Taemin Lee',
      author_email='persuade@gmail.com',
      description='Transformers library for KoGPT2',
      packages=find_packages(exclude=['distillation', 'subtask']),
      long_description=open('../README.md', 'r', encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      python_requires='>=3',
      zip_safe=False,
      include_package_data=True,
      classifiers=(
          'Programming Language :: Python :: 3.6',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ),
      install_requires=require_packages,
      keywords="kogpt2 pytorch transformers"
      )
