from setuptools import setup, find_packages


setup(name='jupy',
      version='0.2',
      packages=find_packages(),
      scripts=['jupy'],
      install_requires=['jupyterlab>=1.2.3', 'argparse>=1.1'],
      author="Kynon Benjamin & Apuã Paquola",
      author_email="kj.benjamin90@gmail.com",
      decription="Starts a jupyter-lab session within host at directory provided.",
      url="https://github.com/KrotosBenjamin/erwin_paquola/tree/master/jupy",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GPLv3 License",
      ],
      zip_safe=False)
