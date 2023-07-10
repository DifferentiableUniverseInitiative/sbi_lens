from setuptools import setup, find_packages

setup(
    name='sbi_lens',
    version='0.0.1',
    url='https://github.com/DifferentiableUniverseInitiative/sbi_lens',
    description=
    'Weak lensing differentiable simulator for Likelihood-free inference applications',
    packages=find_packages(),
    package_dir={'sbi_lens': 'sbi_lens'},
    package_data={
        'sbi_lens': ['data/*.csv', 'data/*.npy', 'data/*.pkl'],
    },
    include_package_data=True,
    install_requires=[
        'numpy>=1.22.4,<1.24',
        'jax>=0.4.1',
        'tensorflow_probability>=0.19.0',
        'dm-haiku>=0.0.9',
        'jaxopt>=0.6',
        'numpyro>=0.10.1',
        'jax-cosmo>=0.1.0',
        'lenstools>=1.2',
        'astropy>=5.2.2',
        'optax>=0.1.4',
        'scikit-learn>=1.2.0',
        'wheel'
    ],
)
