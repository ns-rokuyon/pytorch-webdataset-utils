from setuptools import setup


if __name__ == '__main__':
    setup(name='webdatasetutils',
          version='0.1.0',
          author='S. Nishimura',
          author_email='ness3psi@gmail.com',
          packages=['webdatasetutils'],
          package_dir={'webdatasetutils': 'webdatasetutils'},
          install_requires=[
              'webdataset>=0.1.49'
          ]
    )
