## Retrospective Adversarial Replay for Continual Learning (NeurIPS 2022)


## Abstract

Continual learning is an emerging research challenge in machine learning that addresses the problem where models quickly fit the most recently trained-on data and are prone to catastrophic forgetting due to distribution shifts --- it does this by maintaining a small historical replay buffer in replay-based methods. To avoid these problems, this paper proposes a method, ``Retrospective Adversarial Replay (RAR)'', that synthesizes adversarial samples near the forgetting boundary. RAR perturbs a buffered sample towards its nearest neighbor drawn from the current task in a latent representation space. By replaying such samples, we are able to refine the boundary between previous and current tasks, hence combating forgetting and reducing bias towards the current task.  To mitigate the severity of a small replay buffer, we develop a novel MixUp-based strategy to increase replay variation by replaying mixed augmentations.  Combined with RAR, this achieves a holistic framework that helps to alleviate catastrophic forgetting. We show that this excels on broadly-used benchmarks and outperforms other continual learning baselines especially when only a small buffer is used. We conduct a thorough ablation study over each key component as well as a hyperparameter sensitivity analysis to demonstrate the effectiveness and robustness of RAR.

#### Code coming soon!

## Questions?

For RAR related questions, contact [Lilly](lkumari@uw.edu)  </br>
