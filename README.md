# DRPO: Double Robust Preference Optimization

## Experiments

### IMDb

**DR vs. IS vs. DM**

right $\hat g$: provided by dpo-style preference model,  i.e. $\hat g(y_1,y_2,x) = \exp\left(\beta\log\frac{\pi(y_1|x)}{\pi_\mathrm{ref}(y_1|x)} - \beta\log\frac{\pi(y_2|x)}{\pi_\mathrm{ref}(y_2|x)}\right)$

- Note, the model is not guaranteed to be correctly specified, but we perhaps need to avoid using oracle g such that DM may be the best estimator of the value function $v(\pi_\theta)$

wrong $\hat g$, reverse the original preference, i.e. $\hat g = 1 - \hat g$

right $\pi_\mathrm{ref}$: the SFT that generated the training and validation dataset

wrong $\pi_\mathrm{ref}$: the base-model of SFT (pretrained yet not supervised-fine-tuned)



complementary:

*reward given different KL-divergence level*

<DPO, PPO needed>



### TL;DR

SFT: `cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr` or `trl-lib/pythia-1b-deduped-tldr-sft`

DPO:

PPO: 

DRPO-bt:

- reward model: `cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr`

DRPO-gpm: 

- general preference model:

NashMD:



*prompt for judging*:



### HH: Dialogue

SFT:

DPO:

PPO:

DRPO-bt:

DRPO-gpm:

NashMD:



*prompt for judging*:
