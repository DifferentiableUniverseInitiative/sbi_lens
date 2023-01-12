from setuptools import setup, find_packages

setup(
  name='sbi_lens',
  version='0.0.1',
  url='https://github.com/DifferentiableUniverseInitiative/sbi_lens',
  description='Weak lensing differentiable simulator for Likelihood-free inference applications',
  packages=find_packages(),
  package_dir={'sbi_lens':'sbi_lens'}, 
  package_data={
      'sbi_lens': ['data/*.csv', 'data/*.npy', 'data/*.pkl'],
   },
  include_package_data=True,
  install_requires=[
    'numpy>=1.19.2',
    'jax>=0.2.0',
    'tensorflow_probability>=0.14.1',
    'scikit-learn>=0.21',
    'jaxopt>=0.2',
    'numpyro>=0.9.0',
  ],
)

