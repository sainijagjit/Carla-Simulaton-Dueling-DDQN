# Carla-Simulato-Dueling-DDQN

<h2><b>Dueling Double Deep Q-Network (Dueling DDQN)</b></h2>
In the dueling variant of the DQN, we incorporate an intermediate layer in the Q-Network to estimate both the state value and the state-dependent advantage function. After reformulation (see ref), it turns out we can express the estimated Q-Value as the state value, to which we add the advantage estimate and subtract its mean. This factorization of state-independent and state-dependent values helps disentangling learning across actions and yields better results.
<br><br>
Trained a model to overtake front vehicle in carla environment taking image as input. 
State is formed by stacking last 4 images.
<br>
[![Watch the video](https://github.com/sainijagjit/Carla-Simulaton-Dueling-DDQN/blob/master/Screenshot%20from%202020-07-02%2020-07-33.png)](https://drive.google.com/file/d/1q-IK11GlPLRgP1JlCBtKgLT0Z2U96_GB/view?usp=sharing)

<h1><b>Requirements</b></h1><br>
The follow packages are required, you can install them with pip3 install [package]
opencv-python<br>
gym<br>
gym[atari]<br>
tensorflow<br>
keras<br>
scipy
