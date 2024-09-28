
# Retrospective Adversarial Replay for Continual Learning (NeurIPS 2022)

## Abstract

Continual learning is an emerging research challenge in machine learning that addresses the problem where models quickly fit the most recently trained-on data and are prone to catastrophic forgetting due to distribution shifts --- it does this by maintaining a small historical replay buffer in replay-based methods. To avoid these problems, this paper proposes a method, ``Retrospective Adversarial Replay (RAR)'', that synthesizes adversarial samples near the forgetting boundary. RAR perturbs a buffered sample towards its nearest neighbor drawn from the current task in a latent representation space. By replaying such samples, we are able to refine the boundary between previous and current tasks, hence combating forgetting and reducing bias towards the current task.  To mitigate the severity of a small replay buffer, we develop a novel MixUp-based strategy to increase replay variation by replaying mixed augmentations.  Combined with RAR, this achieves a holistic framework that helps to alleviate catastrophic forgetting. We show that this excels on broadly-used benchmarks and outperforms other continual learning baselines especially when only a small buffer is used. We conduct a thorough ablation study over each key component as well as a hyperparameter sensitivity analysis to demonstrate the effectiveness and robustness of RAR.


## Methods covered:

- **ER**: Random experience replay. Set `args.method` to `rand_replay`.
- **ER-mix**: Random experience replay with mixup applied to the replay samples. Set `args.method` to `er_mixup`.
- **ER-RAR**: RAR (Retrieval-Augmented Replay) using ER for memory retrieval. Set `args.method` to `er_rar`.
- **ER-mix-RAR**: RAR applied to mixup samples retrieved using ER. Set `args.method` to `er_mixup_rar`.

- **MIR**: Maximally Interfered Replay. Set `args.method` to `mir_replay`.
- **MIR-mix**: MIR with mixup applied to replay samples. Set `args.method` to `mir_mixup`.
- **MIR-RAR**: RAR using MIR for memory retrieval. Set `args.method` to `mir_rar`.
- **MIR-mix-RAR**: RAR applied to mixup samples retrieved using MIR. Set `args.method` to `mir_mixup_rar`.

- **ASER**: Adversarial Shapley Value Experience Replay. Set `args.method` to `aser_replay`.
- **ASER-mix**: ASER with mixup applied to replay samples. Set `args.method` to `aser_mixup`.
- **ASER-RAR**: RAR using ASER as the memory retrieval method. Set `args.method` to `aser_rar`.
- **ASER-mix-RAR**: RAR applied to mixup samples retrieved using ASER. Set `args.method` to `aser_mixup_rar`.


## Sample commands for different datasets: 

**Split-CIFAR10 (Memory size 200):**

```bash
python main.py --method mir_rar --dataset split_cifar10 --subsample 50 --batch_size 10 --buffer_batch_size 10 --mem_size 20 --lr 0.1 --n_runs 15 --disc_iters 1 --samples_per_task -1  --device 0 --suffix 'mir_rar' --d_momentum --d_coeff 0.5 --d_eps 0.0314 --d_alpha 0.0314 --d_steps 2

python main.py --method mir_mixup_rar --dataset split_cifar10 --subsample 50 --batch_size 10 --buffer_batch_size 10 --mem_size 20 --lr 0.1 --n_runs 15 --disc_iters 1 --samples_per_task -1  --device 0 --suffix 'mir_mixup_rar' --d_momentum --d_coeff 0.1 --d_eps 0.0314 --d_alpha 0.0314 --d_steps 2
```

**Split-CIFAR10 (Memory size 500):**

```bash
python main.py --method mir_rar --dataset split_cifar10 --subsample 50 --batch_size 10 --buffer_batch_size 10 --mem_size 50 --lr 0.1 --n_runs 15 --disc_iters 1 --samples_per_task -1  --device 0 --suffix 'mir_rar' --d_momentum --d_coeff 0.1 --d_eps 0.0314 --d_alpha 0.0314 --d_steps 2

python main.py --method mir_mixup_rar --dataset split_cifar10 --subsample 50 --batch_size 10 --buffer_batch_size 10 --mem_size 50 --lr 0.1 --n_runs 15 --disc_iters 1 --samples_per_task -1  --device 0 --suffix 'mir_mixup_rar' --d_momentum --d_coeff 0.075 --d_eps 0.0314 --d_alpha 0.0314 --d_steps 2
```

**Split-MNIST (Memory size 200):**

```bash
python main.py --method mir_rar --dataset split_mnist --subsample 50 --batch_size 10 --buffer_batch_size 10 --mem_size 20 --lr 0.1 --n_runs 20 --disc_iters 1 --samples_per_task 1000  --device 0 --suffix 'mir_rar' --d_momentum --d_coeff 0.3 --d_eps 0.3 --d_alpha 0.075 --d_steps 5

python main.py --method mir_mixup_rar --dataset split_mnist --subsample 50 --batch_size 10 --buffer_batch_size 10 --mem_size 20 --lr 0.1 --n_runs 20 --disc_iters 1 --samples_per_task 1000  --device 0 --suffix 'mir_mixup_rar' --d_momentum --d_coeff 0.3 --d_eps 0.3 --d_alpha 0.075 --d_steps 5
```

**Split-MNIST (Memory size 500):**

```bash
python main.py --method mir_rar --dataset split_mnist --subsample 50 --batch_size 10 --buffer_batch_size 10 --mem_size 50 --lr 0.1 --n_runs 20 --disc_iters 1 --samples_per_task 1000  --device 0 --suffix 'mir_rar' --d_momentum --d_coeff 0.3 --d_eps 0.3 --d_alpha 0.075 --d_steps 5

python main.py --method mir_mixup_rar --dataset split_mnist --subsample 50 --batch_size 10 --buffer_batch_size 10 --mem_size 50 --lr 0.1 --n_runs 20 --disc_iters 1 --samples_per_task 1000  --device 0 --suffix 'mir_mixup_rar' --d_momentum --d_coeff 0.4 --d_eps 0.3 --d_alpha 0.075 --d_steps 5
```


## Acknowledgements
We sincerely thank the authors of the following repositories for sharing their code publicly
- MIR: https://github.com/optimass/Maximally_Interfered_Retrieval
- ASER: https://github.com/RaptorMai/online-continual-learning


## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@inproceedings{kumari2022retrospective,
    title={Retrospective adversarial replay for continual learning},
    author={Kumari, Lilly and Wang, Shengjie and Zhou, Tianyi and Bilmes, Jeff A},
    booktitle={Advances in Neural Information Processing Systems},
    pages={28530--28544},
    volume={35},
    year={2022}    
}
```


## Questions?

For RAR related questions, contact [Lilly](lkumari@uw.edu)  </br>