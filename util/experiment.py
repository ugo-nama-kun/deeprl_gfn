import logging
import os
import statistics
import time
from collections import deque

import numpy as np
import pfrl

from pfrl.experiments.evaluator import save_agent, record_stats


def _run_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    scores = []
    lengths = []
    intero_errs = []
    food_eatens = []
    terminate = False
    timestep = 0

    reset = True
    while not terminate:
        if reset:
            obs = env.reset()
            done = False
            test_r = 0
            episode_len = 0
            intero_err = None
            food_num = None
            info = {}
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        test_r += r
        episode_len += 1

        # COMMENT: Update this for other error metrics. Euclid norm is used now
        interoception = info.get("interoception")
        if interoception:
            err = np.linalg.norm(interoception, 2)
            if intero_err is None:
                intero_err = err
            else:
                intero_err += err

        food_eaten = info.get("food_eaten")
        if food_eaten is not None:
            if food_num is None:
                food_num = np.zeros(2)
            else:
                food_num += np.array(food_eaten, dtype=np.uint)

        timestep += 1
        reset = done or episode_len == max_episode_len or info.get("needs_reset", False)
        agent.observe(obs, r, done, reset)
        if reset:
            logger.info(
                "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
            )
            # As mixing float and numpy float causes errors in statistics
            # functions, here every score is cast to float.
            scores.append(float(test_r))
            lengths.append(float(episode_len))
            intero_errs.append(float(intero_err))
            food_eatens.append(food_num)
        if n_steps is None:
            terminate = len(scores) >= n_episodes
        else:
            terminate = timestep >= n_steps
    # If all steps were used for a single unfinished episode
    if len(scores) == 0:
        scores.append(float(test_r))
        lengths.append(float(episode_len))
        intero_errs.append(float(intero_err) if intero_err else intero_err)
        food_eatens.append(food_num)
        logger.info(
            "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
        )

    food_eaten_blue = [food[0] for food in food_eatens]
    food_eaten_red = [food[1] for food in food_eatens]
    return scores, lengths, intero_errs, food_eaten_blue, food_eaten_red


def run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    with agent.eval_mode():
        return _run_episodes(
            env=env,
            agent=agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )


def _batch_run_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns in a batch manner."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_intero_errs = dict()
    episode_food_eaten = dict()
    episode_indices = np.zeros(num_envs, dtype="i")
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_len = np.zeros(num_envs, dtype="i")
    episode_intero_e = np.zeros(num_envs, dtype=np.float64)
    episode_food_num = np.zeros((num_envs, 2), dtype=np.uint)

    obss = env.reset()
    rs = np.zeros(num_envs, dtype="f")

    termination_conditions = False
    timestep = 0
    while True:
        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        episode_r += rs
        episode_len += 1

        # COMMENT: Update this for other error metrics. Euclid norm is used now
        for env_index, info in enumerate(infos):
            interoception = info.get("interoception")
            if interoception is None:
                # Set nan if there is not interoception in info
                episode_intero_e[env_index] = None
            else:
                err = np.linalg.norm(interoception, 2)
                episode_intero_e[env_index] += err

            food_eaten = info.get("food_eaten")
            if food_eaten is not None:
                episode_food_num[env_index] += np.array(food_eaten, dtype=np.uint)

        # Compute mask for done and reset
        if max_episode_len is None:
            resets = np.zeros(num_envs, dtype=bool)
        else:
            resets = episode_len == max_episode_len
        resets = np.logical_or(
            resets, [info.get("needs_reset", False) for info in infos]
        )

        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]

                # time-average of the interoception error
                episode_intero_errs[episode_indices[index]] = episode_intero_e[index] / episode_len[index]

                # food eaten
                episode_food_eaten[episode_indices[index]] = episode_food_num[index]

                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        episode_r[end] = 0
        episode_len[end] = 0
        episode_intero_e[end] = 0
        episode_food_num[end] = np.zeros(2)

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        eval_episode_intero_errs = []
        eval_episode_food_eatens = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
                    eval_episode_intero_errs.append(episode_intero_errs[index])
                    eval_episode_food_eatens.append(episode_food_eaten[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = np.where(
                    episode_indices == first_unfinished_episode
                )[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(episode_r[unfinished_index])
                        eval_episode_lens.append(episode_len[unfinished_index])
                        eval_episode_intero_errs.append(episode_intero_errs[unfinished_index])
                        eval_episode_food_eatens.append(episode_food_eaten[unfinished_index])

        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
                    eval_episode_intero_errs.append(episode_intero_errs[index])
                    eval_episode_food_eatens.append(episode_food_eaten[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.
        agent.batch_observe(obss, rs, dones, resets)

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret, epi_intero_e, epi_food_eaten) in enumerate(
        zip(eval_episode_lens, eval_episode_returns, eval_episode_intero_errs, eval_episode_food_eatens)
    ):
        logger.info(f"evaluation episode {i} length: {epi_len} R: {epi_ret} intero error: {epi_intero_e} food eaten (b, r): {epi_food_eaten}")
    scores = [float(r) for r in eval_episode_returns]
    lengths = [float(ln) for ln in eval_episode_lens]
    intero_errors = [err for err in eval_episode_intero_errs]
    food_eaten_blue = [food[0] for food in eval_episode_food_eatens]
    food_eaten_red = [food[1] for food in eval_episode_food_eatens]
    return scores, lengths, intero_errors, food_eaten_blue, food_eaten_red


def batch_run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns in a batch manner.

    Args:
        env (VectorEnv): Environment used for evaluation.
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of total timesteps to evaluate the agent.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes
            longer than this value will be truncated.
        logger (Logger or None): If specified, the given Logger
            object will be used for logging results. If not
            specified, the default logger of this module will
            be used.

    Returns:
        List of returns of evaluation runs.
    """
    with agent.eval_mode():
        return _batch_run_episodes(
            env=env,
            agent=agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )


def eval_performance(
    env, agent, n_steps, n_episodes, max_episode_len=None, logger=None
):
    """Run multiple evaluation episodes and return statistics.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation episodes.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        Dict of statistics.
    """

    assert (n_steps is None) != (n_episodes is None)

    if isinstance(env, pfrl.env.VectorEnv):
        scores, lengths, intero_errs, food_eaten_blue, food_eaten_red = batch_run_evaluation_episodes(
            env,
            agent,
            n_steps,
            n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )
    else:
        scores, lengths, intero_errs, food_eaten_blue, food_eaten_red = run_evaluation_episodes(
            env,
            agent,
            n_steps,
            n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )
    stats = dict(
        episodes=len(scores),
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        max=np.max(scores),
        min=np.min(scores),
        length_mean=statistics.mean(lengths),
        length_median=statistics.median(lengths),
        length_stdev=statistics.stdev(lengths) if len(lengths) >= 2 else 0,
        length_max=np.max(lengths),
        length_min=np.min(lengths),
        intero_error=np.mean(intero_errs),
        blue_eaten=np.mean(food_eaten_blue),
        red_eaten=np.mean(food_eaten_red),
    )
    return stats


def get_columns(agent, env):
    # Columns that describe information about an experiment.
    basic_columns = (
        "steps",  # number of time steps taken (= number of actions taken)
        "episodes",  # number of episodes finished
        "elapsed",  # time elapsed so far (seconds)
        "mean",  # mean of returns of evaluation runs
        "median",  # median of returns of evaluation runs
        "stdev",  # stdev of returns of evaluation runs
        "max",  # maximum value of returns of evaluation runs
        "min",  # minimum value of returns of evaluation runs
        "length_mean",  # mean of the episode length of evaluation runs
        "length_median",  # median of the episode length of evaluation runs
        "length_stdev",  # stdev of the episode length of evaluation runs
        "length_max",  # max of the episode length of evaluation runs
        "length_min",  # min of the episode length of evaluation runs
        "intero_error_mean",  # Average error of the interoeption
        "blue_eaten",  # Average of eaten blue food
        "red_eaten", # Average of eaten red food
    )

    custom_columns = tuple(t[0] for t in agent.get_statistics())
    env_get_stats = getattr(env, "get_statistics", lambda: [])
    assert callable(env_get_stats)

    custom_env_columns = tuple(t[0] for t in env_get_stats())
    column_names = basic_columns + custom_columns + custom_env_columns

    return column_names


def write_header(outdir, agent, env):
    column_names = get_columns(agent, env)

    with open(os.path.join(outdir, "scores.txt"), "w") as f:
        print("\t".join(column_names), file=f)


class Evaluator(object):
    """Object that is responsible for evaluating a given agent.

    Args:
        agent (Agent): Agent to evaluate.
        env (Env): Env to evaluate the agent on.
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean of returns in evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
    """

    def __init__(
            self,
            agent,
            env,
            n_steps,
            n_episodes,
            eval_interval,
            outdir,
            log_interval,
            max_episode_len=None,
            step_offset=0,
            evaluation_hooks=(),
            save_best_so_far_agent=True,
            logger=None,
    ):
        assert (n_steps is None) != (n_episodes is None), (
                "One of n_steps or n_episodes must be None. "
                + "Either we evaluate for a specified number "
                + "of episodes or for a specified number of timesteps."
        )
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.log_interval = log_interval
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.prev_eval_t = self.step_offset - self.step_offset % self.eval_interval
        self.evaluation_hooks = evaluation_hooks
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)
        self.env_get_stats = getattr(self.env, "get_statistics", lambda: [])
        self.env_clear_stats = getattr(self.env, "clear_statistics", lambda: None)
        assert callable(self.env_get_stats)
        assert callable(self.env_clear_stats)

        # Write a header line first
        write_header(self.outdir, self.agent, self.env)

    def evaluate_and_update_max_score(self, t, episodes):
        self.env_clear_stats()
        eval_stats = eval_performance(
            self.env,
            self.agent,
            self.n_steps,
            self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger,
        )
        elapsed = time.time() - self.start_time
        agent_stats = self.agent.get_statistics()
        custom_values = tuple(tup[1] for tup in agent_stats)
        env_stats = self.env_get_stats()
        custom_env_values = tuple(tup[1] for tup in env_stats)
        mean = eval_stats["mean"]
        values = (
                (
                    t,
                    episodes,
                    elapsed,
                    mean,
                    eval_stats["median"],
                    eval_stats["stdev"],
                    eval_stats["max"],
                    eval_stats["min"],
                    eval_stats["length_mean"],
                    eval_stats["length_median"],
                    eval_stats["length_stdev"],
                    eval_stats["length_max"],
                    eval_stats["length_min"],
                    eval_stats["intero_error"],
                    eval_stats["blue_eaten"],
                    eval_stats["red_eaten"],
                )
                + custom_values
                + custom_env_values
        )
        record_stats(self.outdir, values)

        data_names = get_columns(self.agent, self.env)

        data_dict = dict(zip(data_names, values))

        for hook in self.evaluation_hooks:
            hook(
                env=self.env,
                agent=self.agent,
                evaluator=self,
                step=t,
                eval_stats=eval_stats,
                agent_stats=agent_stats,
                env_stats=env_stats,
            )

        # USE BEST SURVIVAL AGENT
        score = eval_stats["length_mean"]
        if score >= self.max_score:
            self.logger.info("The best score is updated %s -> %s", self.max_score, score)
            self.max_score = score
            if self.save_best_so_far_agent:
                save_agent(self.agent, "best", self.outdir, self.logger)
        return score

    def evaluate_if_necessary(self, t, episodes):
        if t >= self.prev_eval_t + self.eval_interval:
            score = self.evaluate_and_update_max_score(t, episodes)
            self.prev_eval_t = t - t % self.eval_interval
            return score
        return None


def train_agent_batch(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=None,
        log_interval=None,
        max_episode_len=None,
        step_offset=0,
        evaluator=None,
        successful_score=None,
        step_hooks=(),
        return_window_size=100,
        logger=None,
        no_done_at_end=False,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        outdir (str): Path to the directory to output things.
        checkpoint_freq (int): frequency at which agents are stored.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    Returns:
        List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)
    recent_returns = deque(maxlen=return_window_size)
    recent_episode_lens = deque(maxlen=return_window_size)

    num_envs = env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype="i")
    episode_len = np.zeros(num_envs, dtype="i")
    prev_time = time.time()

    # Initial save if checkpoint is enabled
    if checkpoint_freq:
        save_agent(agent, 0, outdir, logger, suffix="_checkpoint")

    # o_0, r_0
    obss = env.reset()

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    save_times = [300000 * 300, 300000 * 400, 300000 * 500, np.iinfo(np.int32).max]
    inter_save_time = save_times.pop(0)

    eval_stats_history = []  # List of evaluation episode stats dict
    try:
        while True:
            # a_t
            actions = agent.batch_act(obss)
            # o_{t+1}, r_{t+1}
            obss, rs, dones, infos = env.step(actions)
            episode_r += rs
            episode_len += 1

            # Compute mask for done and reset
            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = episode_len == max_episode_len
            resets = np.logical_or(
                resets, [info.get("needs_reset", False) for info in infos]
            )

            # if use reset instead of done
            if no_done_at_end:
                resets = np.array(dones).copy()
                dones = np.zeros_like(dones, dtype=bool)

            # Agent observes the consequences
            agent.batch_observe(obss, rs, dones, resets)

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            # For episodes that ends, do the following:
            #   1. increment the episode count
            #   2. record the return
            #   3. clear the record of rewards
            #   4. clear the record of the number of steps
            #   5. reset the env to start a new episode
            # 3-5 are skipped when training is already finished.
            episode_idx += end
            recent_returns.extend(episode_r[end])
            recent_episode_lens.extend(episode_len[end])

            for _ in range(num_envs):
                t += 1
                if checkpoint_freq and t % checkpoint_freq == 0:
                    save_agent(agent, t, outdir, logger, suffix="_checkpoint")

                for hook in step_hooks:
                    hook(env, agent, t)

            if (
                    log_interval is not None
                    and t >= log_interval
                    and t % log_interval < num_envs
            ):
                logger.info(
                    ">>> @{:.0f} iter --- step:{} episode:{} last_R: {} average_R:{} running average episode len:{:.0f}".format(
                        # NOQA
                        t / log_interval,
                        t,
                        np.sum(episode_idx),
                        recent_returns[-1] if recent_returns else np.nan,
                        np.mean(recent_returns) if recent_returns else np.nan,
                        np.mean(recent_episode_lens) if recent_episode_lens else np.nan,
                    )
                )
                # logger.info("statistics: {}".format(agent.get_statistics()))
                logger.info("time-interval: {}".format(time.time() - prev_time))

                prev_time = time.time()
            if evaluator:
                eval_score = evaluator.evaluate_if_necessary(
                    t=t, episodes=np.sum(episode_idx)
                )
                if eval_score is not None:
                    eval_stats = dict(agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                    if (
                            successful_score is not None
                            and evaluator.max_score >= successful_score
                    ):
                        break

            if t >= steps:
                break
            if t >= inter_save_time:
                save_agent(agent, t, outdir, logger, suffix="_inter")
                inter_save_time = save_times.pop(0)

            # Start new episodes if needed
            episode_r[end] = 0
            episode_len[end] = 0
            obss = env.reset(not_end)

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        env.close()
        if evaluator:
            evaluator.env.close()
        raise
    else:
        # Save the final model
        save_agent(agent, t, outdir, logger, suffix="_finish")

    return eval_stats_history


def train_agent_batch_with_evaluation(
        agent,
        env,
        steps,
        eval_n_steps,
        eval_n_episodes,
        eval_interval,
        outdir,
        no_done_at_end,
        checkpoint_freq=None,
        max_episode_len=None,
        step_offset=0,
        eval_max_episode_len=None,
        return_window_size=100,
        eval_env=None,
        log_interval=None,
        successful_score=None,
        step_hooks=(),
        evaluation_hooks=(),
        save_best_so_far_agent=True,
        logger=None,
):
    """Train an agent while regularly evaluating it.

    Args:
        agent: Agent to train.
        env: Environment train the againt against.
        steps (int): Number of total time steps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        checkpoint_freq (int): frequency with which to store networks
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If set to None, max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        logger (logging.Logger): Logger used in this function.
    Returns:
        agent: Trained agent.
        eval_stats_history: List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent_batch:
            raise ValueError(
                "{} does not support train_agent_batch_with_evaluation().".format(hook)
            )

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        evaluation_hooks=evaluation_hooks,
        save_best_so_far_agent=save_best_so_far_agent,
        logger=logger,
        log_interval=log_interval,
    )

    eval_stats_history = train_agent_batch(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger,
        no_done_at_end=no_done_at_end,
    )

    return agent, eval_stats_history
