# FCSRL

## Installation
1) We recommend to use Anaconda or Miniconda to manage python environment.
2) Create conda env,
    ```shell
    cd FCSRL
    conda env create -f environment.yaml
    conda activate FCSRL
    ```
3) Install PyTorch according to your platform and cuda version.
4) Install FCSRL,
    ```shell
    pip install -e .
    ```

## Training
To run a single experiment, take `PointGoal1` for example, run
```shell
python scripts/{BASE_RL_ALG}_repr_CMDP.py --env_name SafetyPointGoal1Gymnasium-v0 --cudaid 0 --seed 100
```
where `{BASE_RL_ALG}` can be `ppo` or `td3`. For other task, you can simply replace `PointGoal1` and choose a task from `[PointGoal1, PointButton1, PointPush1, PointGoal2, CarGoal1, CarButton1]`.

For image-based task, 
```shell
python scripts/td3_repr_vision_CMDP.py --env_name SafetyPointGoal2Gymnasium-v0 --cudaid 0 --seed 100
```
Note here we still adopt the original env and use a [observation wrapper](./fcsrl/env/gym_utils.py#L48) based to get vision observation to accelerate the training.