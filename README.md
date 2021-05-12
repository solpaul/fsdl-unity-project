# Full Stack Deep Learning Course Project Write-Up

For my FSDL project I decided to learn about the basic theory and application of reinforcement learning in the <a href="https://unity.com/">Unity game engine</a>.

Unity Technologies provide a well-documented toolkit to carry out the training of agents in game environments, as such this was a convenient introduction point to RL for a beginner like me. Despite this, starting from zero knowledge of Unity or RL meant that there was a fair degree of learning curve in getting to successfully training an agent in a unique environment - this report will walk through the journey I took to get there.

## Getting started with Unity and the ML-Agents Toolkit - Installation and Basics

The Unity Machine Learning Agents Toolkit is an open-source project that allows anyone to easily apply state-of-the-art algorithms to agents in game environments. Installation was straightforward and consisted of:

- Installing Unity on my local machine
- Cloning the ML-Agents Toolkit repo to my local machine
- Installing the mlagents Python pacakge from PyPi
- Adding a package to my project within the Unity Game Engine itself (com.unity.ml-agents)

I then ran though <a href="https://github.com/Unity-Technologies/ml-agents/blob/release_17_docs/docs/Getting-Started.md">Getting Started Guide</a> to make sure the installation had completed correctly.

Before starting on ML-Agents I did the introductory tutorials on Unity itself to familiarise myself with the platform. I also watched this YouTube video: <a href="https://www.youtube.com/watch?v=pwZpJzpE2lQ">LEARN UNITY - The Most BASIC TUTORIAL I'll Ever Make</a> which was brilliant for the absolute beginner. 

Armed with my newfound Unity knowledge I moved onto the <a href="https://github.com/Unity-Technologies/ml-agents/blob/release_17_docs/docs/Learning-Environment-Create-New.md">Making a New Learning Environment<a> tutorial. I knew that I wanted to create a simple 3D environment conceptually very similar to this 2D example so it was a good starting point.

## Creating and training an agent in a simple environment

![image](https://user-images.githubusercontent.com/11899284/117721295-9755ce00-b1d7-11eb-9597-51bedcd70008.png)

This simple scene is set up in Unity - a flat platform, a box (the target), and a rolling ball (the agent).

Everything in Unity is controlled with C# scripts, including agents. MLAgents has an <strong>Agent</strong> class that contains all the boilerplate code to run an agent in an environment, we simply inherit from this class and add the following additional behaviours:

- Move the target to a new position at the start of each episode
- Collect observations at some fixed time interval (the positions of target and agent, and the velocity of agent)
- Apply forces to the agent in the x and y directions
- Code to allow us to apply those forces manually using keyboard inputs (helpful for debugging)
- Add rewards when certain things occur

The rewards are what drive the learning algorithm. In this case we give a reward of 1 and end the episode if the agent hits the target (the episode ends if the agent rolls off the platform).

ML-Agents is set up so that training is done separately in Python/PyTorch. This is carried out in a similar way to the FSDL text-recognizer labs, running a Python program through the command line using the mlagents-learn command. The main argument is a yaml config file that contains all the parameters for training, and there are additional optional arguments. I installed the mlagents package in an anaconda environment and kicked off training in this environment by running the following command:

```
mlagents-learn config/rollerball_config.yaml run-id=rollerball_01
```

In RL, agents learn a policy that takes in some state of the environment and outputs actions. In this case the state is the observations described above (8 numbers - x, y and z coordinates for the agent and target, and the x and y components of the agent's velocity), and the actions are forces to apply on the rolling ball (2 numbers - forces in the ball's local x and y directions). How are these inputs mapped to outputs? A neural net. In this case two fully connected layers with 128 hidden units each.

My understanding it that the agent's policy in this case is a neural net with two fully connected layers, with 8 input numbers (x, y and z coordinates for the agent and target, and the x and y components of the agent's velocity), and 2 outputs (forces in the x and y direction).

![vector policy nn](./vector_policy_nn.PNG)

ML-Agents provides Tensorboard integration so I was able to look at training progress for the average reward and episode length.

![tensorboard RollerBall_01](./tensorboard_rollerball_01.PNG)

This was a simple task so it converged to a maximum average reward of 1 in less than 30k steps (10 minutes on my local machine). 

##PPO Algorithm

Part of my goal was to understand more about reinforcement learning, but of course this is a very big subject so I was only able to scratch the surface. Despite this, I did my best to understand the algorithm I used in ML-Agents to train my agents: Proximal Policy Optimization (PPO). PPO is an algorithm developed by OpenAI in 2017 with three broad goals:

- Ease of implementation
- Sample efficient
- Easy to tune

A few words on the later two of these goals. PPO is a policy gradient method, which means it learns directly from whatver the agent experiences (run a batch of sequences, use the rewards to calculate an objective function, back propagate the gradient to update the policy). The issue is that the policy gradient may use the information ineffieciently (for example, an episode which ends in a negative reward but during which the agent actually displayed desireable behaviours) - there is a credit assignment problem in a sparse reward setting. Which behaviours are actually causing the reward? RL algorithms can be very sample inefficient.

The training data is also dependent on the current policy, rather than a static dataset (e.g. with supervised learning), so the data distribution over observations and rewards are constantly changing. This can make it very unstable and sensitive to hyperparameters - for example, choosing a particular learning rate can cause training to fall into a local optima from which it cannot escape, because it cannot generate the data to do so.

PPO seeks to to update
