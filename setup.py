from setuptools import setup, find_packages

setup(name="pfs.datamodel",
      #version="x.y",
      author="",
      #author_email="",
      #description="",
      url="https://github.com/Subaru-PFS/datamodel/",
      packages=find_packages("python"),
      package_dir={'':'python'},
      zip_safe=True,
      license="",
      install_requires=["numpy"],
      )
