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

Rewards:
- Hit target - 1.0

ML-Agents is set up so that training is done separately in Python/PyTorch. This is carried out in a similar way to the FSDL text-recognizer labs, running a Python program through the command line using the mlagents-learn command. The main argument is a yaml config file that contains all the parameters for training, and there are additional optional arguments. I installed the mlagents package in an anaconda environment and kicked off training in this environment by running the following command:

```
mlagents-learn config/rollerball_config.yaml --run-id=rollerball_01
```

In RL, agents learn a policy that takes in some state of the environment and outputs actions. In this case the state is the observations described above (8 numbers - x, y and z coordinates for the agent and target, and the x and y components of the agent's velocity), and the actions are forces to apply on the rolling ball (2 numbers - forces in the ball's local x and y directions). How are these inputs mapped to outputs? A neural net. In this case two fully connected layers with 128 hidden units each.

My understanding it that the agent's policy in this case is a neural net with two fully connected layers, with 8 input numbers (x, y and z coordinates for the agent and target, and the x and y components of the agent's velocity), and 2 outputs (forces in the x and y direction).

![vector policy nn](./vector_policy_nn.PNG)

ML-Agents provides Tensorboard integration so I was able to look at training progress for the average reward and episode length.

![tensorboard RollerBall_01](./tensorboard_rollerball_01.PNG)

This was a simple task so it converged to a maximum average reward of 1 in less than 30k steps (10 minutes on my local potato-powered machine). 

## Some theory - RL and the PPO Algorithm

Part of my goal was to understand more about reinforcement learning, but of course this is a very big subject so I was only able to scratch the surface. Despite this, I did my best to understand the algorithm I used in ML-Agents to train my agents: Proximal Policy Optimization (PPO). PPO is an algorithm developed by OpenAI in 2017 with three broad goals:

- Ease of implementation
- Sample efficient
- Easy to tune

A few words on the later two of these goals. PPO is a policy gradient method, which means it learns directly from whatver the agent experiences (run a batch of sequences, use the rewards to calculate an objective function, back propagate the gradient to update the policy). The issue is that the policy gradient may use the information ineffieciently (for example, an episode which ends in a negative reward but during which the agent actually displayed desireable behaviours) - there is a credit assignment problem in a sparse reward setting. Which behaviours are actually causing the reward? RL algorithms can be very sample inefficient.

The training data is also dependent on the current policy, rather than a static dataset (e.g. with supervised learning), so the data distribution over observations and rewards are constantly changing. This can make it very unstable and sensitive to hyperparameters - for example, choosing a particular learning rate can cause training to fall into a local optima from which it cannot escape, because it cannot generate the data to do so.

PPO seeks to to update

## Moving from 2D to a 3D free body

Expanding this simple problem into three dimensions was a good way to increase the complexity of the problem, while still keeping a similar training and reward structure. To do this I simply added five more planes to create a cube, ensuring they were all facing inwards to contain the agent. I wanted the agent to be operating as if in space i.e. as a free body with no forces acting upon it. This would mean that the agent would need to learn how to accelerate and decelerate itself around the environment (similar to the rolling in two dimensions). To do this was simply a case of turning off gravity in the Unity environment, and allowing collisions with the boundary to be fully elastic. To increase the level of control required I also added a negative reward for the agent hitting the boundary.

Rewards:
- Hit target - 1.0
- Hit boundary - -0.5

Here's what the environment looked like.

![RollerBall 02 Environment](./rollerball_02_environment.PNG)

All I needed to add to the C# script controlling the behaviour of the agent was an additional observation for the velocity of the agent in the z (vertical) direction, an additional force on the agent in the z direction, and the negative reward for a collision with the boundary. In the 2D plane there was an easy way for episodes to end by the agent sphere rolling off the edge, this was not the case now so I added a time limit to each episode of 1000 steps. Finally, since the problem had increased in complexity (there were now 9 inputs and 3 outputs) I took advantage of the ability to run multiple environments simultaneously in Unity, adding a further 7 copies of the agent/target/box.

![RollerBall 02 Multi Environment](./rollerball_02_multi_environment.PNG)

Once again I used the mlagents-learn command to run training, incrementing the run id to keep track of results, and increasing the maximum number of iterations to account for the increased complexity.

```
mlagents-learn config/rollerball_config.yaml --run-id=rollerball_02
```

![tensorboard RollerBall_02](./tensorboard_rollerball_02.PNG)

Training progress is shown here in blue (alongside the previous run in orange), converging to 1 within 150k steps, around five times longer than previously. However, due to running eight parallel environment this only took 15 minutes on my machine, just 50% longer than the first challenge.

## Adding orientation control

The previous tasks employed a spherical agent with no requirement for directional control. The agent only needed to translate its own position close to that of the target. The logical next step was therefore to add a 'face' to the agent, and only allow it to complete the task by colliding this face with the target.

To do this I changed the agent into a cube, and only gave a reward if the front side of this cube collided with the target. I added code to the C# script that gave the agent three degrees of rotational freedom: around its x-axis (pitch), y-axis (yaw), and z-axis (roll). Implementing smooth rotations was tricky, but ultimately I was able to have the agent rotate based on inputs from each of the three directions. This meant that there were now an additional three outputs on top of the existing three.

![BlueAgent Environment](./blueagent_environment.PNG)

This change made an enormous difference to the difficulty of the task and I initially spent a lot of time trying and failing to train the agent. First, I experimented with the observations provided to the agent:

- Positions of agent and target, agent velocity (as previous) - unsuccessful
- Added agent's current rotation - unsuccessful
- Added agent-to-target vector (center to center) - unsuccessful
- Changed agent-to-target vector (face of agent to target surface) - unsuccessful
- Changed agent-to-target vector (center of agent to target surface, normalised) - unsucessful
- Added magnitude of vector (i.e. distance) from agent to target - unsuccesful
- Added agent's current angular velocity - partially successful

This experimentation was not entirely wasted since it allowed me to observe improvements in the agent's ability to control its own position/orientation, and eventually achieve some success in the task. However, it's overall performance was poor, nowhere near the average rewards achieved in the previous tasks.

At this point I put some thought into the other side of the problem, the controls. I realised that the roll (rotation around the z-axis) was redundant in terms of the task I was trying to achieve. It didn't matter if the face was 'upside down' upon impact with the target, only that it was facing the target. This meant I could reduce the output space to five numbers (forces in the x, y and z directions; rotations around the x-axis and y-axis) which helped considerably.

Finally, I considered the model and reward system. I wanted the agent to move towards the target as efficiently as possible, so I added a time penalty. The longer the agent took to get to the target, the lower the reward. I increased the capacity of the neural network to 3 layers of 512 units each.

Rewards:

- Hit target - 1.0
- Hit boundary - -0.5
- Time penalty - linearly increasing from 0 at start of episode to -1 at end of episode (after the maximum of 1000 steps)

I increased the capacity of the neural network to 3 layers of 512 units each. 

![tensorboard BlueAgent_02](./tensorboard_blueagent_02.PNG)

As can be seen there was a large increase in training time, 7 million steps (over 3 hours on my local machine). Furthermore, while performance looked reasonable in terms of the average reward, the agent would sometimes lack smoothness in getting to the target. I made one more observation adjustment, changing the agent-to-target vector to the local coordinate system of the agent.

After a significant investment of time, I finally felt that the performance was good enough to move on. 

## Speeding up training - GPUs in cloud

the more than three hours per training run was a limitation on my ability to try different things. I was training on my own laptop without a GPU so I thought I could get a considerable speed-up by using a cloud machine with a GPU. I had not used Azure previously so I signed up for an account to take advantage of the free credits available.

Spinning up an appropriate instance was straightforward. I went for a NC6s_v3 VM with 6 CPUs and a V100 GPU, using Windows to match my own machine. Surprisingly, the increase in speed was minimal. I couldn't quite figure out why this was the case, although reading Unity forums it seems this is an issue many face. Users suggested that significant advantage would only be found for visual observations, so I returned to my local machine and saved the VM for when I started using cameras in my agent.

Ultimately, this meant that I was limited in what I could test. I wanted to try some hyperparameter optimisation, but it was not feasible in the time I had with these extended training times.

## Directional observation using rays

Since I had succesfully trained an agent by giving it a vector to the target as an input, it seemed likely that replacing this vector with a raycast would yeild similar results. Rays in Unity provide information by detecting objects at straight lines from the agent. I created an array of these rays to scan the volume directly in front of the agent, and removed the previous vector observation.

![BlueAgent Rays](./blueagent_rays.PNG)

Initially, my rays were hitting the boundary, but I was unable to successfully train an agent in this way. I made the boundaries invisible to the rays and was able to get the agent to learn.

![tensorboard BlueAgent_09](./tensorboard_blueagent_09.PNG)

As can be seen, the introduction of rays resulted in a huge increase in training time. I ended up training it for 50 million steps over almost 2.5 days! Still, the agent did eventually achieve reasonable performance, the average reward was lower and episode time higher, but this was to be expected since the agent needed to scan the area around it by rotating, which it did so effectively.

## Visual observation

Finally, I was able to try the visual observation functionality in Unity ML-Agents. This incorporates a convnet to process images from a camera on the agent, and feed the output alongside the other observations. To set this up I added a forward facing camera to the agent and included a visual encoder type in the yaml file. I knew that this was going to increase the size of the network considerably, so I kept the image as small as possible at just 32x32 pixels (in RGB in order to ensure the red target could be distinguihed from the background and shadows in the scene).

![BlueAgent Camera](./blueagent_camera.PNG)

As can be seen, this resolution was suffient to distiguish the target. Training was done one last time in the same way.

![tensorboard BlueAgent_09](./tensorboard_blueagent_12.PNG)


## Summary and learnings
