from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='stabilized-lime',
      version='0.1.0',
      description='Stabilized-LIME for Model Explanation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ZhengzeZhou/slime',
      author='Zhengze Zhou',
      author_email='zz433@cornell.edu',
      license='BSD',
      packages=find_packages(exclude=['js', 'node_modules', 'tests']),
      python_requires='>=3.5',
      install_requires=[
          'lime',
          'matplotlib',
          'numpy',
          'scipy',
          'tqdm >= 4.29.1',
          'scikit-learn>=0.18',
          'scikit-image>=0.12',
          'pyDOE2==1.3.0'
      ],
      extras_require={
          'dev': ['pytest', 'flake8'],
      },
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      include_package_data=True,
      zip_safe=False)
