from setuptools import setup, find_packages

requires_list = [] #TODO

setup(name='musculoco_il',
      version='0.1',
      description='Imitation Learning for Musculoskeltal Control of Huamnoid Locomotion',
      packages=[package for package in find_packages()
                if package.startswith('musculoco_il')],
      install_requires=requires_list,
      zip_safe=False,
      )
