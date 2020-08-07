#!/usr/bin/python3

import functools;
from absl import app;
import tensorflow as tf;
from tf_agents.drivers import dynamic_episode_driver; # data collection driver
from tf_agents.environments import tf_py_environment, suite_mujoco; # environment and problem
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment; # multiple workers
from tf_agents.system import system_multiprocessing as multiprocessing; # every worker works on a signle process
from tf_agents.metrics import tf_metrics; # all kinds of metrics
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork;
from tf_agents.networks.value_rnn_network import ValueRnnNetwork; # network structure
from tf_agents.agents.ppo import ppo_agent; # ppo agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer; # replay buffer
from tf_agents.policies import policy_saver; # random policy
from tf_agents.utils import common; # element_wise_squared_loss

batch_size = 64; # how many workers

def main(_):

  # environment serves as the dataset in reinforcement learning
  train_env = tf_py_environment.TFPyEnvironment(ParallelPyEnvironment([lambda: suite_mujoco.load('HalfCheetah-v2')] * batch_size));
  eval_env = tf_py_environment.TFPyEnvironment(suite_mujoco.load('HalfCheetah-v2'));
  # create agent
  actor_net = ActorDistributionRnnNetwork(train_env.observation_spec(), train_env.action_spec(), lstm_size = (100, 100));
  value_net = ValueRnnNetwork(train_env.observation_spec());
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 1e-3);
  tf_agent = ppo_agent.PPOAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    optimizer = optimizer,
    actor_net = actor_net,
    value_net = value_net,
    normalize_observations = False,
    normalize_rewards = False,
    use_gae = True,
    num_epochs = 25
  );
  tf_agent.initialize();
  # replay buffer
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    tf_agent.collect_data_spec,
    batch_size = train_env.batch_size,
    max_length = 1000000
  );
  # policy saver
  saver = policy_saver.PolicySaver(tf_agent.policy);
  # define trajectory collector
  train_episode_count = tf_metrics.NumberOfEpisodes();
  train_total_steps = tf_metrics.EnvironmentSteps();
  train_avg_reward = tf_metrics.AverageReturnMetric(batch_size = train_env.batch_size);
  train_avg_episode_len = tf_metrics.AverageEpisodeLengthMetric(batch_size = train_env.batch_size);
  train_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    train_env,
    tf_agent.collect_policy, # rollout policy
    observers = [
      replay_buffer.add_batch,
      train_episode_count,
      train_total_steps,
      train_avg_reward,
      train_avg_episode_len
    ], # callbacks when an episode is completely collected
    num_episodes = 30, # how many episodes are collected in an iteration
  );
  # training
  eval_avg_reward = tf_metrics.AverageReturnMetric(buffer_size = 30);
  eval_avg_episode_len = tf_metrics.AverageEpisodeLengthMetric(buffer_size = 30);
  while train_total_steps.result() < 25000000:
    train_driver.run();
    trajectories = replay_buffer.gather_all();
    loss, _ = tf_agent.train(experience = trajectories);
    replay_buffer.clear(); # clear collected episodes right after training
    if tf_agent.train_step_counter.numpy() % 50 == 0:
      print('step = {0}: loss = {1}'.format(tf_agent.train_step_counter.numpy(), loss));
    if tf_agent.train_step_counter.numpy() % 500 == 0:
      # save checkpoint
      saver.save('checkpoints/policy_%d' % tf_agent.train_step_counter.numpy());
      # evaluate the updated policy
      eval_avg_reward.reset();
      eval_avg_episode_len.reset();
      eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        eval_env,
        tf_agent.policy,
        observers = [
          eval_avg_reward,
          eval_avg_episode_len,
        ],
        num_episodes = 30, # how many epsiodes are collected in an iteration
      );
      eval_driver.run();
      print('step = {0}: Average Return = {1} Average Episode Length = {2}'.format(tf_agent.train_step_counter.numpy(), train_avg_reward.result(), train_avg_episode_len.result()));
  # play cartpole for the last 3 times and visualize
  import cv2;
  for _ in range(3):
    status = eval_env.reset();
    policy_state = tf_agent.policy.get_initial_state(eval_env.batch_size);
    while not status.is_last():
      action = tf_agent.policy.action(status, policy_state);
      status = eval_env.step(action.action);
      policy_state = action.state;
      cv2.imshow('halfcheetah', eval_env.pyenv.envs[0].render());
      cv2.waitKey(25);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  multiprocessing.handle_main(functools.partial(app.run, main));
