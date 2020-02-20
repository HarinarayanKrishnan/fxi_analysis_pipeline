from distutils.core import setup
setup(name='icxray',
      version='1.0',
      description='Code for X-ray nano-tomography of ICs',
      author='Michael Sutherland',
      author_email='michael.sutherland@dmea.osd.mil',
      install_requires=['psutil','numpy','scipy','tomopy','mpi4py'],
      packages=['icxray'],
      )
