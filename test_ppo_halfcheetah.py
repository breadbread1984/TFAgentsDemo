#!/usr/bin/python3

import tensorflow as tf;
from tf_agents.environments import tf_py_environment, suite_mujoco; # environment
from tf_agents.policies import policy_saver; # policy
import cv2;

def main():

  # environment
  eval_env = tf_py_environment.TFPyEnvironment(suite_mujoco.load('HalfCheetah-v2'));
  # deserialize saved policy
  saved_policy = tf.compat.v2.saved_model.load('checkpoints/policy_1000/');
  # apply_policy and visualize
  total_return = 0.0;
  for _ in range(10):
    episode_return = 0.0;
    status = eval_env.reset();
    policy_state = saved_policy.get_initial_state(eval_env.batch_size);
    while not status.is_last():
      action = saved_policy.action(status, policy_state);
      status = eval_env.step(action.action);
      policy_state = action.state;
      cv2.imshow('halfcheetah', eval_env.pyenv.envs[0].render());
      cv2.waitKey(25);
      episode_return += status.reward;
    total_return += episode_return;
  avg_return = total_return / 10;
  print("average return is %f" % avg_return);

if __name__ == "__main__":

  main();

