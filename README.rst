.. -*- mode: rst -*-

=========
Case-based Policy Inference (CBPI) for Transfer in Reinforcement Learning
=========

|License|_

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. _License: https://github.com/cowhi/CBPI/blob/master/LICENSE.txt


.. image:: https://github.com/cowhi/CBPI/raw/master/results_ECML/cbr_rl.png
  :alt: Experiment setup
  :width: 654
  :height: 400
  :align: center


Description
============

This is the accompanying code for the paper **Case-based Policy Inference
for Transfer in Reinforcement Learning** submitted to ECML 2017.

Also accepted as a short paper **Case-based Policy Inference** at the TiRL
workshop at AAMAS 2017.

Run an experiment
-----------------

Just download the repository, check the experiment settings in
**params_CBPI.yaml** and use the shell to start the experiment:

.. code:: shell

    python experiment_CBPI.py

A log-folder will be created and the whole experiment is documented and the
graphs are created. If you choose visual, you also get a video of the testing
episodes (needs much space).

Some results
============

Comparing **CBPI** to Probabilistic Policy Reuse **PLPR** and **Q-Learning**:

.. image:: https://github.com/cowhi/CBPI/raw/master/results_ECML/compare_reward_mean_12345.png
  :alt: Experiment setup
  :width: 548
  :height: 200
  :align: center

.. image:: https://github.com/cowhi/CBPI/raw/master/results_ECML/compare_reward_mean_234.png
  :alt: Experiment setup
  :width: 548
  :height: 200
  :align: center

References
==========

1. Aamodt, A., Plaza, E.: **Case-based reasoning: Foundational issues, methodological variations, and system approaches.** AI communications 7(1), 39–59 (1994)
2. Aha, D.W.: **The omnipresence of case-based reasoning in science and application.** Knowledge-based systems 11(5), 261–273 (1998)
3. Bianchi, R.A., Ros, R., De Mantaras, R.L.: **Improving reinforcement learning by using case based heuristics.** In: Proceedings of the 8th International Conference on Case-Based Reasoning (ICCBR). pp. 75–89. Springer (2009)
4. Bridle, J.S.: **Training stochastic model recognition algorithms as networks can lead to maximum mutual information estimation of parameters.** In: Proceedings of the 2nd International Conference on Neural Information Processing Systems. pp. 211– 217. MIT Press (1989)
5. Cheetham, W., Watson, I.: **Fielded applications of case-based reasoning.** The Knowledge Engineering Review 20(03), 321–323 (2005)
6. Fernandez, F., Veloso, M.: **Probabilistic policy reuse in a reinforcement learning agent.** In: Proceedings of the 5th International Conference on Autonomous Agents and Multiagent Systems (AAMAS). pp. 720–727 (2006)
7. Gabel, T., Riedmiller, M.: **Cbr for state value function approximation in reinforcement learning.** In: Proceedings of the 6th International Conference on Case-Based Reasoning (ICCBR). pp. 206–221. Springer (2005)
8. Glatt, R., da Silva, F.L., Costa, A.H.R.: **Towards knowledge transfer in deep reinforcement learning.** In: Proceedings of the 5th Brazilian Conference on Intelligent Systems (BRACIS). pp. 91–96. IEEE (2016)
9. Hullermeier, E.: **Credible case-based inference using similarity profiles.** IEEE Transactions on Knowledge and Data Engineering 19(6), 847–858 (2007)
10. Koga, M.L., Freire, V., Costa, A.H.: **Stochastic abstract policies: Generalizing knowledge to improve reinforcement learning.** Cybernetics, IEEE Transactions on 45(1), 77–88 (2015)
11. Kolodner, J.: **Case-based reasoning.** Morgan Kaufmann (2014)
12. Konidaris, G., Scheidwasser, I., Barto, A.G.: **Transfer in reinforcement learning via shared features.** Journal of Machine Learning Research (JMLR) 13(1), 1333–1371 (2012)
13. Kuhlmann, G., Stone, P.: **Graph-based domain mapping for transfer learning in general games.** In: Proceedings of the 18th European Conference in Machine Learning (ECML). pp. 188–200. Springer (2007)
14. McCallum, R.A.: **Instance-based utile distinctions for reinforcement learning with hidden state.** In: Proceedings of the 12th International Conference on Machine Learning (ICML). pp. 387–395 (1995)
15. Mnih, V., Silver, D., Rusu, A.A., Riedmiller, M., et al.: **Human-level control through deep reinforcement learning.** Nature 518(7540), 529–533 (2015)
16. Ng, A.Y., Coates, A., Diel, M., Ganapathi, V., Schulte, J., Tse, B., Berger, E., Liang, E.: **Autonomous inverted helicopter flight via reinforcement learning.** In: Experimental Robotics IX, pp. 363–372. Springer (2006)
17. Pan, S.J., Yang, Q.: **A survey on transfer learning.** Knowledge and Data Engineering, IEEE Transactions on 22(10), 1345–1359 (2010)
18. Puterman, M.L.: **Markov decision processes: discrete stochastic dynamic programming.** John Wiley & Sons, Hoboken, NJ, USA (2014)
19. Sharma, M., Holmes, M.P., Santamarıa, J.C., Irani, A., Isbell Jr, C.L., Ram, A.: **Transfer learning in real-time strategy games using hybrid cbr/rl.** In: Proceedings of the 20th International Joint Conference on Artificial Intelligence (IJCAI). vol. 7, pp. 1041–1046 (2007)
20. Sherstov, A.A., Stone, P.: **Function approximation via tile coding: Automating parameter choice.** In: Abstraction, Reformulation and Approximation, pp. 194– 205. Springer (2005)
21. Silva, F.L.d., Glatt, R., Costa, A.H.R.: **Simultaneously learning and advising in multiagent reinforcement learning.** In: Proceedings of the 16th International Conference on Autonomous Agents and Multiagent Systems (AAMAS) (2017)
22. Sinapov, J., Narvekar, S., Leonetti, M., Stone, P.: **Learning inter-task transferability in the absence of target task samples.** In: Proc. 14th International Conference on Autonomous Agents and Multiagent Systems (AAMAS). pp. 725–733 (2015)
23. Stone, P., Sutton, R.S.: **Scaling reinforcement learning toward robocup soccer.** In: Proceedings of the 18th International Conference of Machine Learning (ICML). pp. 537–544. ACM (2001)
24. Sutton, R.S., Barto, A.G.: **Introduction to Reinforcement Learning.** MIT Press, Cambridge, MA, USA (1998)
25. Taylor, M.E., Stone, P.: **Transfer learning for reinforcement learning domains: A survey.** Journal of Machine Learning Research (JMLR) 10, 1633–1685 (2009)
26. Tesauro, G.: **Temporal difference learning and td-gammon.** Communications of the ACM 38(3), 58–68 (1995)
27. Thrun, S., Mitchell, T.M.: **Lifelong robot learning**, vol. 15. Elsevier (1995)
28. Thrun, S., Schwartz, A.: **Finding structure in reinforcement learning.** Proceedings of the 7th International Conference on Neural Information Processing Systems (NIPS-94) pp. 385–392 (1995)
29. Watkins, C.J., Dayan, P.: **Q-learning.** Machine Learning 8(3-4), 279–292 (1992)
30. Watson, I.: **Case-based reasoning is a methodology not a technology.** Knowledge-based systems 12(5), 303–308 (1999)
