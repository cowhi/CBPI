.. -*- mode: rst -*-

=========
Case-based Policy Inference (CBPI)
=========

|License|_ |Docs|_

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. _License: https://github.com/cowhi/CBPI/blob/master/LICENSE.txt


.. image:: https://github.com/cowhi/CBPI/blob/master/results_ecml/cbr_rl.png
  :alt: Experiment setup
  :width: 740
  :height: 435
  :align: center


Description
============

This is the accompanying code for the paper **Case-based Policy Inference
for Transfer in Reinforcement Learning** submitted to ECML 2017.

Also accepted as a short paper **Case-based Policy Inference** at the TiRL 
workshop at AAMAS 2017.

Run an experiment
--------------------------

Just download the repository, check the experiment settings in
**params_CBPI.yaml** and use the shell to start the experiment:

.. code:: shell

    python experiment_CBPI.py

A log-folder will be created and the whole experiment is documented and the
graphs are created. If you choose visual, you also get a video of the testing
episodes (needs much space).
