from setuptools import setup, find_packages


setup(name='jupy',
      version='0.2',
      packages=find_packages(),
      scripts=['jupy'],
      install_requires=[
          'jupyterlab>=1.2.3',
          'argparse>=1.1'
      ],
      author="Kynon Benjamin & Apuã Paquola",
      author_email="kj.benjamin90@gmail.com",
      decription="A utility to launch jupyter lab remotely.",
      package_data={
          '': ['*org'],
      },
      url="https://github.com/KrotosBenjamin/erwin_paquola/tree/master/jupy",
      license='GPLv3',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3",
      ],
      keywords='jupyter jupyterlab ssh remote',
      zip_safe=False)
