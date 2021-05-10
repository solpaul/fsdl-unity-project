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

