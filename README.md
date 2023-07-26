# lstm_explore
## "Visual Episodic Memory based Exploration".  A convolutional LSTM autoencoder curiosity method for robot exploration.

This work explores the use of visual episodic memory as a 
source of intrinsic motivation for robotic exploration 
problems. Using a convolutional recurrent
neural network autoencoder, the agent learns an efficient
representation for spatiotemporal features such that accurate
sequence prediction can only happen once spatiotemporal fea-
tures have been learned. Structural similarity between ground
truth and autoencoder generated images is used as an intrinsic
motivation signal to guide exploration. Our proposed episodic
memory model also implicitly accounts for the agent’s actions,
motivating the robot to seek new interactive experiences rather
than just areas that are visually dissimilar. When guiding
robotic exploration, our proposed method outperforms the
Curiosity-driven Variational Autoencoder (CVAE) at finding
dynamic anomalies.

![image](https://user-images.githubusercontent.com/4203994/195624929-d562f43e-faee-4cef-8f34-11f428c8f094.png)

## Getting Started
In addition to the requirements.txt, this experiment requires ROS(Noetic) and Gazebo installation.  This configuration was tested on Ubuntu 20.04 with two Nvidia GPU's and CUDA 11.2.  The ROS_files directory contains the Gazebo map as well as Turtlebot configuration files.  lstmAE.py is the primary convolutional LSTM exploration method with vae.py being the variational autoencoder method for comparison.

### Install ROS Noetic with Gazebo
https://wiki.ros.org/noetic/Installation/Ubuntu

### Install dependencies
`$ pip install -r requirements.txt`

### ROS commands
```
roslaunch turtlebot3_gazebo turtlebot3_curiosity.launch
roslaunch turtlebot3_slam turtlebot3_slam.launch
roslaunch turtlebot3_navigation move_base.launch
roslaunch explore_lite explore.launch
```

### Then run python script
`python3 lstmAE.py`
