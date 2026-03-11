# Downloaded Papers

## Loss Landscape Geometry - Foundational

1. **Visualizing the Loss Landscape of Neural Nets** (li2018_visualizing_loss_landscape.pdf)
   - Authors: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein
   - Year: 2018 (NeurIPS)
   - arXiv: 1712.09913
   - Why relevant: Introduces filter normalization for meaningful loss landscape visualization; establishes correlation between landscape flatness and generalization; shows skip connections produce smoother landscapes.

2. **Sharpness-Aware Minimization for Efficiently Improving Generalization** (foret2020_sharpness_aware_minimization.pdf)
   - Authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
   - Year: 2021 (ICLR)
   - arXiv: 2010.01412
   - Why relevant: Defines SAM objective connecting neighborhood-wise loss sharpness to generalization via PAC-Bayes bounds; provides theoretical and empirical framework for flat minima seeking.

3. **Sharp Minima Can Generalize for Deep Nets** (dinh2017_sharp_minima_can_generalize.pdf)
   - Authors: Laurent Dinh, Razvan Pascanu, Samy Bengio, Yoshua Bengio
   - Year: 2017 (ICML)
   - arXiv: 1703.04933
   - Why relevant: Shows that naive sharpness measures (Hessian norm, volume flatness) can be manipulated via reparametrization; essential methodological warning for loss landscape studies.

4. **An Investigation into Neural Net Optimization via Hessian Eigenvalue Density** (ghorbani2019_investigation_hessian.pdf)
   - Authors: Behrooz Ghorbani, Shankar Krishnan, Ying Xiao
   - Year: 2019 (ICML)
   - arXiv: 1901.09588
   - Why relevant: Characterizes Hessian spectrum structure (bulk + outliers); shows outlier eigenvalues correspond to per-class gradient components.

5. **Empirical Analysis of the Hessian of Over-Parametrized Neural Networks** (sagun2017_empirical_hessian.pdf)
   - Authors: Levent Sagun, Utku Evci, V. Ugur Guney, Yann Dauphin, Leon Bottou
   - Year: 2017
   - arXiv: 1706.04454
   - Why relevant: Foundational analysis of Hessian spectrum in overparameterized networks; discovers that most eigenvalues cluster near zero.

6. **Emergent Properties of the Local Geometry of Neural Loss Landscapes** (fort2019_emergent_geometry.pdf)
   - Authors: Stanislav Fort, Surya Ganguli
   - Year: 2019
   - arXiv: 1906.04724
   - Why relevant: Studies local curvature properties, gradient covariance structure, and their relationship to network architecture.

## Mode Connectivity & Topology

7. **Git Re-Basin: Merging Models modulo Permutation Symmetries** (ainsworth2022_git_rebasin.pdf)
   - Authors: Samuel K. Ainsworth, Jonathan Hayase, Siddhartha Srinivasa
   - Year: 2023 (ICLR)
   - arXiv: 2209.04836
   - Why relevant: Demonstrates that loss landscapes contain nearly a single basin after accounting for permutation symmetries; key evidence for landscape connectivity.

8. **The Role of Permutation Invariance in Linear Mode Connectivity** (entezari2021_permutation_invariance.pdf)
   - Authors: Rahim Entezari, Hanie Sedghi, Olga Saukh, Behnam Neyshabur
   - Year: 2021
   - arXiv: 2110.06296
   - Why relevant: Conjectures that SGD solutions are linearly mode connected modulo permutation symmetries; foundational for understanding basin structure.

9. **Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness** (zhao2020_bridging_mode_connectivity.pdf)
   - Authors: Pu Zhao, Pin-Yu Chen, Payel Das, Karthikeyan Natesan Ramamurthy, Xue Lin
   - Year: 2020
   - arXiv: 1912.05671
   - Why relevant: Connects mode connectivity to adversarial robustness; provides methods for finding low-loss paths between minima.

10. **Explaining Landscape Connectivity of Low-cost Solutions for Multilayer Nets** (kuditipudi2019_explaining_connectivity.pdf)
    - Authors: Rohith Kuditipudi, Xiang Wang, Holden Lee, Yi Zhang, Zhiyuan Li, Wei Hu, Sanjeev Arora, Rong Ge
    - Year: 2019 (NeurIPS)
    - arXiv: 1906.06247
    - Why relevant: Proves that dropout-stable optima are connected via low-loss paths; provides theoretical foundation for mode connectivity.

11. **Deep Ensembles: A Loss Landscape Perspective** (fort2019_deep_ensembles_landscape.pdf)
    - Authors: Stanislav Fort, Huiyi Hu, Balaji Lakshminarayanan
    - Year: 2019
    - arXiv: 2002.11642
    - Why relevant: Shows that deep ensemble diversity arises from exploring different basins/modes in the loss landscape.

12. **Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling** (benton2021_loss_surface_simplexes.pdf)
    - Authors: Gregory Benton, Wesley Maddox, Sanae Lotfi, Andrew Gordon Wilson
    - Year: 2021 (ICML)
    - arXiv: 2012.09874
    - Why relevant: Extends mode connectivity from paths to volumes; shows SGD solutions form connected low-loss volumes.

13. **Taxonomizing Local versus Global Structure in Neural Network Loss Landscapes** (simsek2021_taxonomizing_local_global.pdf)
    - Authors: Berfin Simsek, François Ged, Arthur Jacot, Francesco Spadaro, Clément Hongler, Wulfram Gerstner, Johanni Brea
    - Year: 2021
    - arXiv: 2107.12356
    - Why relevant: Provides systematic taxonomy of local (curvature) vs global (connectivity) landscape properties.

14. **Using Mode Connectivity for Loss Landscape Analysis** (gotmare2018_mode_connectivity_analysis.pdf)
    - Authors: Akhilesh Gotmare, Nitish Shirish Keskar, Caiming Xiong, Richard Socher
    - Year: 2018
    - arXiv: 1806.11484
    - Why relevant: Uses mode connectivity as a tool for understanding training dynamics and optimizer behavior.

## Neural Theorem Proving

15. **Generative Language Modeling for Automated Theorem Proving** (polu2020_gpt_theorem_proving.pdf)
    - Authors: Stanislas Polu, Ilya Sutskever
    - Year: 2020
    - arXiv: 2009.03393
    - Why relevant: GPT-f; foundational work applying transformer LMs to theorem proving with step-by-step tactic generation.

16. **HyperTree Proof Search for Neural Theorem Proving** (lample2022_hypertree.pdf)
    - Authors: Guillaume Lample, Timothee Lacroix, Marie-Anne Lachaux, et al.
    - Year: 2022 (NeurIPS)
    - arXiv: 2205.11491
    - Why relevant: MCTS-based proof search for step-by-step proving; online training that iteratively improves the model.

17. **Proof Artifact Co-training for Theorem Proving with Language Models** (han2021_proof_artifact_cotraining.pdf)
    - Authors: Jesse Michael Han, Jason Rute, Yuhuai Wu, Edward W. Ayers, Stanislas Polu
    - Year: 2021
    - arXiv: 2102.06203
    - Why relevant: PACT method for co-training on proof artifacts; shows how training data composition affects proving capability.

18. **Baldur: Whole-Proof Generation and Repair with Large Language Models** (first2023_baldur.pdf)
    - Authors: Emily First, Markus N. Rabe, Talia Ringer, Yuriy Brun
    - Year: 2023 (ICSE)
    - arXiv: 2303.04910
    - Why relevant: KEY PAPER for direct proof generation approach; generates entire proofs in one shot without search.

19. **DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** (xin2024_deepseek_prover.pdf)
    - Authors: Huajian Xin et al.
    - Year: 2024
    - arXiv: 2405.14333
    - Why relevant: Shows importance of synthetic data for training neural theorem provers.

20. **DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via RL for Subgoal Decomposition** (ren2025_deepseek_prover_v2.pdf)
    - Authors: Z. Ren, Zhihong Shao et al.
    - Year: 2025
    - arXiv: 2504.21801
    - Why relevant: KEY PAPER for hybrid CoT approach; uses chain-of-thought subgoal decomposition + RL with consistency rewards.

21. **LEGO-Prover: Neural Theorem Proving with Growing Libraries** (xin2023_lego_prover.pdf)
    - Authors: Huajian Xin et al.
    - Year: 2023
    - arXiv: 2310.00656
    - Why relevant: Growing skill library approach; modular proof construction.

22. **NaturalProver: Grounded Mathematical Proof Generation with Language Models** (jiang2022_naturalprover.pdf)
    - Authors: Sean Welleck, Jiacheng Liu, Ximing Lu, Hannaneh Hajishirzi, Yejin Choi
    - Year: 2022
    - arXiv: 2205.12615
    - Why relevant: Natural language proof generation with grounding; represents pure CoT informal reasoning approach.

## Sharpness/Flatness Theory

23. **Towards Understanding Sharpness-Aware Minimization** (andriushchenko2022_understanding_sam.pdf)
    - Authors: Maksym Andriushchenko, Nicolas Flammarion
    - Year: 2022 (ICML)
    - arXiv: 2206.09150
    - Why relevant: Theoretical analysis of why SAM works; connects to implicit bias of optimization.

24. **Asymmetric Valleys: Beyond Sharp and Flat Local Minima** (he2019_asymmetric_valleys.pdf)
    - Authors: Haowei He, Gao Huang, Yang Yuan
    - Year: 2019 (NeurIPS)
    - arXiv: 1907.04595
    - Why relevant: Shows that valleys are asymmetric, not simply sharp/flat; enriches the geometric vocabulary.

25. **A Modern Look at the Relationship between Sharpness and Generalization** (wen2023_modern_sharpness_generalization.pdf)
    - Authors: Maksym Andriushchenko et al.
    - Year: 2023
    - arXiv: 2302.11834
    - Why relevant: Comprehensive re-evaluation of sharpness-generalization relationship with modern methods.

## Specialized Systems

26. **Solving Olympiad Geometry without Human Demonstrations** (trinh2024_alphageometry.pdf)
    - Authors: Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, Thang Luong
    - Year: 2024 (Nature)
    - arXiv: 2401.12880
    - Why relevant: AlphaGeometry; neuro-symbolic approach combining language model with symbolic deduction engine.

## Surveys

27. **A Survey on Deep Learning for Theorem Proving** (lu2024_survey_deep_learning_theorem.pdf)
    - Authors: Zhaoyu Li et al.
    - Year: 2024
    - arXiv: 2404.09939
    - Why relevant: Comprehensive survey of neural theorem proving landscape.

28. **Formal Mathematical Reasoning: A New Frontier in AI** (lu2024_formal_mathematical_reasoning.pdf)
    - Authors: Kaiyu Yang et al.
    - Year: 2024
    - arXiv: 2404.01515
    - Why relevant: Survey of formal mathematical reasoning approaches and challenges.
