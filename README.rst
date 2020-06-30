Rebalance
=========

|Build status| |Coverage| |Code Factor| 

EpiMod is a data-driven epidemiological modeling tool written in Python. More specifically, it models the reported daily number of new COVID-19 infections in a region by coupling the epidemiological SEIR equations with Bayesian inference. A high-level explanation of the tool is given `here <https://siavashadpey.github.io/projects/covid19_projections/>`_. 

.. raw:: html

        <div class="ui container">

        <h2 class="ui dividing header">Installation</h2>

                <div class="ui text container">
.. raw:: html

                    <h3 class="ui header">Clone the repository:</h3>

.. code-block:: bash

    git clone https://github.com/siavashadpey/epimod.git

.. raw:: html

                    <h3 class="ui header">Install the package:</h3>

.. code-block:: bash

    cd epimod
    pip3 install .



.. |Build Status| image:: https://travis-ci.org/siavashadpey/epimod.svg?branch=master
    :target: https://travis-ci.org/siavashadpey/epimod.svg?branch=master
    
.. |Coverage| image:: https://coveralls.io/repos/github/siavashadpey/epimod/badge.svg?branch=master
    :target: https://coveralls.io/repos/github/siavashadpey/epimod/badge.svg?branch=master

.. |Code Factor| image:: https://www.codefactor.io/repository/github/siavashadpey/epimod/badge
   :target: https://www.codefactor.io/repository/github/siavashadpey/epimod