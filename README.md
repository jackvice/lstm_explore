# lstm_explore
"Visual Episodic Memory based Exploration".  A convolutional LSTM autoencoder curiosity method for robot exploration.

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
