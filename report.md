# Udacity Deep Reinforcement Learning 
# Navigation Project report

Ben Hosken

## Implementation




## Learning algorithm


### Model architecture


```
QNetwork(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=128, bias=True)
  (out): Linear(in_features=128, out_features=4, bias=True)
)
```

### Hyper parameters

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


## Rewards

![results](raw_scores_dqn.jpg)

![results](average_scores_dqn.jpg)


```
Episode 100	Average Score: 0.95
Episode 200	Average Score: 4.54
Episode 300	Average Score: 7.58
Episode 400	Average Score: 10.24
Episode 500	Average Score: 11.68
Episode 580	Average Score: 13.00
Environment solved in 480 episodes!	Average Score: 13.00
```

## Ideas for future work

