#!/usr/bin/python3

import tensorflow as tf;
from tf_agents.environments import tf_py_environment, suite_gym; # environment and problem
from tf_agents.policies import policy_saver; # deserialize policy
import cv2; # visualization

def main():

    # environment
    eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'));
    # deserialize saved policy
    saved_policy = tf.compat.v2.saved_model.load('checkpoints/policy_20000/');
    # apply policy and visualize
    total_return = 0.0;
    for _ in range(10):
        status = eval_env.reset();
        episode_return = 0.0;
        while not status.is_last():
            action = saved_policy.action(status);
            status = eval_env.step(action.action);
            cv2.imshow('cartpole', eval_env.pyenv.envs[0].render());
            cv2.waitKey(25);
            episode_return += status.reward;
        total_return += episode_return;
    avg_return = total_return / 10;
    print("average return is %f" % avg_return);

if __name__ == "__main__":

    main();
