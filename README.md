
# ROMA: Multi-Agent Reinforcement Learning with Emergent Roles

## Note
 This codebase accompanies the paper submission "ROMA: Multi-Agent Reinforcement Learning with Emergent Roles" ([ROMA website](https://sites.google.com/view/romarl)), and is based on  [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced.

The implementation of the following methods can also be found in this codebase, which are finished by the authors of [PyMARL](https://github.com/oxwhirl/pymarl):

- [**ROMA**: ROMA: Multi-Agent Reinforcement Learning with Emergent Roles](https://arxiv.org/abs/2003.08039)
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

If you want to run the environments we designed, move all the SC2 maps in `src/envs/starcraft2/map/designed/` to `3rdparty/StarCraftII/Maps/SMAC_Maps/`.
It is worth noting that `bane_vs_bane1` corresponds to `6z4b`, `zb_vs_sz` corresponds to `10z5b_vs_2s3z`, and `sz_vs_zb` 
corresponds to `6s4z_vs_10b30z` in the paper. 

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
python3 src/main.py 
--config=qmix_smac_latent
--env-config=sc2
with
agent=latent_ce_dis_rnn
env_args.map_name=MMM2
t_max=20050000
```

To test other maps, add parameters

```shell
h_loss_weight=5e-2
var_floor=1e-4
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`


All results will be stored in the `Results` folder.



## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.
