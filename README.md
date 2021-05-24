# DQN

![DDQN CarRacing](results/videos/carracing_good.gif)

This is a basic implementation of Deep Q Learning. We have implemented linear and convolutional DQN and DDQN models, with 
DQN and double DQN algorithms

## To run
To begin, setup [OpenAI gym](https://gym.openai.com/) and install the packages in `requirements.txt`.

Run `python -m examples.box2d_ddqn` in the top-level directory.

To run the car racing for human control, 'python car_drrive.py' in the top-level directory.

## Results
The best models trained on each env are present in `results/models/`. There you will find the saved pytorch model as a `.pth` file and
a graph comparing the reward per episode against random play
