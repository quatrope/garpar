==========================
Installation
==========================

You can also `install garpar from PyPI`_ using pip:

.. code-block:: bash

   $ pip install garpar

Finally, you can also install the latest development version of
garpar `directly from GitHub`_:

However you can install the development version from `directly from GitHub`_:

.. code-block:: bash

   $ pip install git+https://github.com/quatrope/garpar/

This is useful if there is some feature that you want to try, but we did
not release it yet as a stable version. Although you might find some
unpolished details, these development installations should work without
problems.

If you find any problem, please open an issue in the `issue tracker`_.

.. warning::

   It is recommended that you
   **never ever use sudo** with distutils, pip, setuptools and friends in Linux
   because you might seriously break your system
   [`1 <http://wiki.python.org/moin/CheeseShopTutorial#Distutils_Installation>`_]
   [`2 <http://stackoverflow.com/questions/4314376/how-can-i-install-a-python-egg-file/4314446#comment4690673_4314446>`_]
   Use `virtual environments <https://docs.python.org/3/library/venv.html>`_ instead.

.. _issue tracker: https://github.com/quatrope/garpar/issues
.. _install garpar from PyPI: https://pypi.python.org/pypi/garpar/
.. _directly from GitHub: https://github.com/quatrope/garpar/


If you don't have Python
-------------------------

If you don't already have a python installation with numpy and scipy, we
recommend to install either via your package manager or via a python bundle.
These come with numpy, scipy, matplotlib and many other helpful
scientific and data processing libraries.
