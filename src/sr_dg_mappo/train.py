#!/usr/bin/env python
"""Train DG-MAPPO-style algorithms on LongRangePredatorPreyContinuous-v0."""

import os
import random
import socket
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

try:
    import setproctitle
except ImportError:
    setproctitle = None

import mat.envs.long_range_predator_prey  # noqa: F401
from mat.config import get_config
from mat.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from mat.envs.long_range_predator_prey import LongRangePredatorPreyTorchVecEnv


def seed_everything(seed, deterministic=False):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_env_kwargs(all_args, seed):
    return {
        "num_predators": all_args.num_predators,
        "num_prey": all_args.num_prey,
        "world_size": all_args.world_size,
        "dt": all_args.env_dt,
        "episode_length": getattr(all_args, "env_episode_length", all_args.episode_length),
        "random_start_positions": all_args.random_start_positions,
        "init_min_predator_dist": all_args.init_min_predator_dist,
        "init_min_prey_dist": all_args.init_min_prey_dist,
        "init_min_prey_predator_dist": all_args.init_min_prey_predator_dist,
        "obs_radius": all_args.obs_radius,
        "comm_radius": all_args.comm_radius,
        "ensure_connected_comm_graph": all_args.ensure_connected_comm_graph,
        "ensure_prey_visible": all_args.ensure_prey_visible,
        "capture_radius": all_args.capture_radius,
        "capture_k": all_args.capture_k,
        "predator_max_speed": all_args.predator_max_speed,
        "predator_max_omega": all_args.predator_max_omega,
        "prey_speed_ratio": all_args.prey_speed_ratio,
        "prey_max_omega": all_args.prey_max_omega,
        "prey_avoid_radius": all_args.prey_avoid_radius,
        "collision_radius": all_args.collision_radius,
        "collision_resolution_iters": all_args.collision_resolution_iters,
        "device": all_args.env_device,
        "seed": seed,
    }


def seed_env(env, seed):
    target = getattr(env, "unwrapped", env)
    if hasattr(target, "seed"):
        target.seed(seed)
    else:
        env.reset(seed=seed)
    for space_name in ("action_space", "observation_space", "share_observation_space"):
        space = getattr(target, space_name, None)
        if hasattr(space, "seed"):
            space.seed(int(seed))


def env_device_uses_cuda(all_args):
    return str(getattr(all_args, "env_device", "cpu")).lower().startswith("cuda")


def make_vec_env(env_fns, num_threads, all_args, label):
    if num_threads == 1:
        return ShareDummyVecEnv(env_fns)

    return ShareSubprocVecEnv(env_fns)


def make_batched_predator_prey_env(all_args, num_envs, seed, label):
    if all_args.env_name != "long_range_predator_prey":
        raise NotImplementedError(f"Unsupported env_name: {all_args.env_name}")
    kwargs = make_env_kwargs(all_args, seed)
    kwargs["num_envs"] = int(num_envs)
    print(
        f"predator-prey: using batched torch VecEnv on {all_args.env_device} "
        f"for {num_envs} {label} rollout envs."
    )
    return LongRangePredatorPreyTorchVecEnv(**kwargs)


def optional_wandb(use_wandb):
    if use_wandb:
        import wandb

        return wandb

    try:
        import wandb
    except ModuleNotFoundError:
        wandb = types.ModuleType("wandb")
        wandb.run = None
        wandb.log = lambda *args, **kwargs: None
        wandb.Image = lambda image, *args, **kwargs: image

        class _NoOpTable:
            def __init__(self, *args, **kwargs):
                self.data = []

            def add_data(self, *args):
                self.data.append(args)

        wandb.Table = _NoOpTable
        sys.modules["wandb"] = wandb

    return wandb


def make_predator_prey_env(all_args, seed):
    seed_everything(seed, deterministic=False)
    env = gym.make(
        all_args.scenario,
        disable_env_checker=True,
        **make_env_kwargs(all_args, seed),
    )
    env = env.unwrapped
    seed_env(env, seed)
    return env


def make_train_env(all_args):
    if env_device_uses_cuda(all_args):
        seed_everything(all_args.seed, deterministic=all_args.cuda_deterministic)
        return make_batched_predator_prey_env(
            all_args,
            all_args.n_rollout_threads,
            all_args.seed,
            "training",
        )

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "long_range_predator_prey":
                raise NotImplementedError(f"Unsupported env_name: {all_args.env_name}")
            return make_predator_prey_env(all_args, all_args.seed + rank * 1000)

        return init_env

    env_fns = [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
    return make_vec_env(env_fns, all_args.n_rollout_threads, all_args, "training")


def make_eval_env(all_args):
    if env_device_uses_cuda(all_args):
        seed_everything(all_args.seed * 50000, deterministic=all_args.cuda_deterministic)
        return make_batched_predator_prey_env(
            all_args,
            all_args.n_eval_rollout_threads,
            all_args.seed * 50000,
            "eval",
        )

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "long_range_predator_prey":
                raise NotImplementedError(f"Unsupported env_name: {all_args.env_name}")
            return make_predator_prey_env(all_args, all_args.seed * 50000 + rank * 10000)

        return init_env

    env_fns = [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
    return make_vec_env(env_fns, all_args.n_eval_rollout_threads, all_args, "eval")


def parse_args(args, parser):
    parser.set_defaults(
        env_name="long_range_predator_prey",
        algorithm_name="sr_mappo",
        experiment_name="predator_prey",
        use_wandb=False,
        n_quants=1,
    )
    parser.add_argument("--scenario", type=str, default="LongRangePredatorPreyContinuous-v0")
    parser.add_argument("--num_predators", type=int, default=6)
    parser.add_argument("--num_prey", type=int, default=2)
    parser.add_argument("--world_size", type=float, default=6.0)
    parser.add_argument("--env_dt", type=float, default=0.1)
    parser.add_argument("--random_start_positions", action="store_true", default=False)
    parser.add_argument("--init_min_predator_dist", type=float, default=0.45)
    parser.add_argument("--init_min_prey_dist", type=float, default=0.45)
    parser.add_argument("--init_min_prey_predator_dist", type=float, default=1.0)
    parser.add_argument(
        "--env_episode_length",
        type=int,
        default=None,
        help="Predator-prey max simulation steps before env timeout. Defaults to --episode_length.",
    )
    parser.add_argument("--obs_radius", type=float, default=1.8)
    parser.add_argument("--comm_radius", type=float, default=2.2)
    parser.add_argument(
        "--disable_connected_comm_graph",
        action="store_false",
        dest="ensure_connected_comm_graph",
        default=True,
    )
    parser.add_argument(
        "--disable_prey_visibility_guarantee",
        action="store_false",
        dest="ensure_prey_visible",
        default=True,
    )
    parser.add_argument("--capture_radius", type=float, default=0.35)
    parser.add_argument("--capture_k", type=int, default=2)
    parser.add_argument("--predator_max_speed", type=float, default=0.22)
    parser.add_argument("--predator_max_omega", type=float, default=2.84)
    parser.add_argument("--prey_speed_ratio", type=float, default=0.85)
    parser.add_argument("--prey_max_omega", type=float, default=2.3)
    parser.add_argument("--prey_avoid_radius", type=float, default=2.4)
    parser.add_argument("--collision_radius", type=float, default=0.18)
    parser.add_argument("--collision_resolution_iters", type=int, default=4)
    parser.add_argument("--env_device", type=str, default="cpu")
    parser.add_argument("--faulty_node", type=int, default=-1)
    parser.add_argument("--eval_faulty_node", type=int, nargs="+", default=[-1])
    parser.add_argument("--add_center_xy", action="store_true", default=False)
    parser.add_argument("--use_state_agent", action="store_true", default=False)
    parser.add_argument("--use_mustalive", action="store_false", default=True)

    all_args = parser.parse_known_args(args)[0]
    if all_args.env_episode_length is None:
        all_args.env_episode_length = all_args.episode_length
    if all_args.eval_faulty_node is None:
        all_args.eval_faulty_node = [-1]
    return all_args


def configure_algorithm(all_args):
    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True
        all_args.truelyDistributed = False

    if all_args.algorithm_name in {"ippo", "consensus_ippo"}:
        all_args.iterations = 0
        all_args.truelyDistributed = False
        all_args.n_quants = 1
        if all_args.algorithm_name == "consensus_ippo":
            all_args.share_policy = False

    if all_args.algorithm_name == "dgn":
        all_args.iterations = 0
        all_args.n_quants = 1

    if all_args.algorithm_name == "mappo_dgnn_dsgd":
        all_args.iterations = max(int(all_args.iterations), 1)
        all_args.truelyDistributed = True
        all_args.n_quants = 1

    if all_args.algorithm_name == "sr_mappo":
        all_args.iterations = max(int(all_args.iterations), 1)
        all_args.truelyDistributed = True
        all_args.n_quants = 1

    if all_args.algorithm_name == "sr_mappo_shared":
        all_args.iterations = max(int(all_args.iterations), 1)
        all_args.truelyDistributed = False
        all_args.n_quants = 1


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    configure_algorithm(all_args)
    seed_everything(all_args.seed, deterministic=all_args.cuda_deterministic)

    wandb = optional_wandb(all_args.use_wandb)

    from mat.runner.shared.ma_gotogoal_runner import MAGoToGoalRunner

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    print("predator-prey config:", all_args)

    results_root = Path(os.environ.get("SR_DG_MAPPO_RESULTS", "results"))
    run_dir = (
        results_root
        / all_args.env_name
        / all_args.scenario
        / f"{all_args.num_predators}pred_{all_args.num_prey}prey"
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.scenario,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
            group=all_args.experiment_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        curr_run = "run1" if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"
        run_dir = run_dir / curr_run
        run_dir.mkdir(parents=True, exist_ok=True)
        run = None

    if setproctitle is not None:
        setproctitle.setproctitle(
            f"{all_args.algorithm_name}-{all_args.env_name}-"
            f"{all_args.experiment_name}@{all_args.user_name}"
        )

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    if all_args.algorithm_name == "dgn":
        raise NotImplementedError(
            "The initial SR-DG-MAPPO port supports mappo_dgnn and sr_mappo."
        )
    runner = MAGoToGoalRunner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        if hasattr(runner.writter, "export_scalars_to_json"):
            runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        elif hasattr(runner.writter, "flush"):
            runner.writter.flush()
        runner.writter.close()


def cli():
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
