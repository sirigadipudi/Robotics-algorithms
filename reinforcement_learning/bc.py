import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50, 
                       batch_size=32):
    
    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy. 
    optimizer = optim.Adam(list(policy.parameters()))
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs)*episode_length // batch_size
    losses = []
    states = []
    actions = []
    for i in range(len(expert_data)):
      states = states + (expert_data[i]['observations'].tolist())
      actions = actions + (expert_data[i]['actions'].tolist())
    for epoch in range(num_epochs): 
        ## TODO Students
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO start: Fill in your behavior cloning implementation here
            obs_data, acts_data = sample_batch(states, actions, batch_size)

            obs_data = torch.tensor(states).float().to(device)
            acts_data = torch.tensor(actions).float().to(device)
            
            a_hat = policy(obs_data)
            loss = torch.nn.functional.mse_loss(a_hat, acts_data)
            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 50 == 0:
            print('[%d] loss: %.8f' %
                (epoch, running_loss / 10.))
        losses.append(loss.item())

def sample_batch(states, actions, batch_size):
  
  indices = np.random.choice(len(states), batch_size, replace=False)
  states = [states[k] for k in indices]
  actions = [actions[k] for k in indices]

  return states, actions