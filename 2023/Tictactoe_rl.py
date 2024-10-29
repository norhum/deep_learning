#unfinished but I want to end it here since things left to do are minor bugs that are unrelated to RL

#tictactoe with me trying to implement reinforcement learning without having any knowledge of RL at all
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

#for GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters and more
lr = 3e-4
num_epochs = 10000
iter = 10000
loss_bin = []
loss_interval = 1000

class Tictactoe:
  def __init__(self):
    self.grid = np.zeros((3,3), dtype=np.int8) #3x3 grid will be implemented as a tensor in the shape of (3,3)
    self.turn = 1
    self.current = False
    self.player = 'A'
    self.grid_list = []
    self.player_list = []
    self.length = 1

  def round(self, input): #input is going to be 0 through 8
    x, y = input//3, input%3

    if self.grid[x, y] != 0:
      print('invalid placement')
      return None
    else:
      self.grid[x, y] = self.turn

    self.grid_list.append(self.grid.tolist())
    self.player_list.append(self.player)
    self.turn *= -1

  def decision(self): #to decide whether it's over or not, returns boolean value or None if it's a draw (for the 1st option)
    if 0 not in self.grid:
      # print(f'Congrats Player {self.player}!')
      self.current = True
      return

    if 3 in [*np.abs(self.grid.sum(0)).tolist(), *np.abs(self.grid.sum(1)).tolist()]:
      # print(f'Congrats Player {self.player}!')
      self.current = True
    elif np.abs(self.grid.trace()) == 3:
      # print(f'Congrats Player {self.player}!')
      self.current = True
    elif np.abs(self.grid[0,2] + self.grid[1,1] + self.grid[2,0]) == 3:
      # print(f'Congrats Player {self.player}!')
      self.current = True
    else:
      self.current = False

  def __call__(self, game):
    i = 0
    while True:
      # user_input = int(input(f"Player {self.player}'s turn "))
      user_input = game[i]
      i += 1
      self.round(user_input)
      self.decision()
      if self.current:
        return self.player
      self.player = 'A' if self.turn == 1 else 'B'
      self.length += 1

#making data input is the grid, output is a boolean value of the final result
class DataMaker:
  def __init__(self, iter=1):
    self.iter = iter
    self.data = []
    self.value = []
    self.length = []

  def game_generator(self):
    return np.random.choice(9, 9, replace=False)

  def __call__(self):
    for _ in range(self.iter):
      game = Tictactoe()
      play = self.game_generator()
      player = game(play)
      if game.length > 5: #only let short games(which are the best games)
        continue

      #for building data
      self.data.append(game.grid_list)
      self.value.append([1 if game.player_list[i] == game.player else 0 for i in range(len(game.grid_list))])
      self.length.append(game.length)

    self.data = torch.tensor([item for sublist in self.data for item in sublist], dtype=torch.float).view(-1, 9).to(device)
    self.value = torch.tensor([item for sublist in self.value for item in sublist], dtype=torch.float).view(-1, 1).to(device)
    return self.data, self.value

b = DataMaker(iter)
b.data, b.value = b()

print(b.data.shape, b.value.shape)

#Tttprediction predicts how likely you will win the game
#outputs the winning accuracy for the player who played the last move of the input grid
class Tttprediction(nn.Module):
  def __init__(self):
    super().__init__()
    self.seq = nn.Sequential(
        nn.Linear(9, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )

  def forward(self, x):
    output = self.seq(x)
    return output
    
#now implement a model that plays tictactoe! (Surprisingly short haha...)
#input is the grid, output is a one hot vector from 0 to 8
class Tttcomputer(nn.Module):
  def __init__(self):
    super().__init__()
    self.seq = nn.Sequential(
        nn.Linear(9, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 9),
    )
  def forward(self, x):
    output = self.seq(x)
    return output

#for Tttprediction
model1 = Tttprediction().to(device)

#loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=lr)

#train 
for i in range(num_epochs):
  #forward
  prediction = model1(b.data)
  loss = criterion(prediction, b.value)

  #backward and optimize
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  #print the loss
  if i % loss_interval == 0:
    loss_bin.append(loss.item())
    print(f'{i//loss_interval}: {loss.item():.4f}')

#for Tttcomputer
model = Tttcomputer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

ans = (b.value==0).nonzero(as_tuple=True)[0].to(device) #torch.Size([3419])
inputs = b.data[ans]
bns = ans+1
oneh = torch.abs(b.data[ans] - b.data[bns])
outputs = (oneh==1).nonzero(as_tuple=True)[1]
#input is the grid when the value is 0, output is the place where 1 has played afterwards

for i in range(len(b.length)):
  if b.length[i] == 5:
    b.length[i] = 2
  elif b.length[i] in [6, 7]:
    b.length[i] = 3
  else:
    b.length[i] = 4

for i in range(num_epochs):
  #forward
  prediction = model(inputs)
  loss = criterion(prediction, outputs)
  
  #backward and optimization
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  #print the loss
  if i % loss_interval == 0:
    loss_bin.append(loss.item())
    print(f'{i//loss_interval}: {loss.item():.4f}')

#draw the loss. Was lazy. Still, it show's both loss
plt.plot(loss_bin)
plt.show()

# for the Tttprediction class
a = model1(torch.tensor([1.,0.,-1.,0.,1.,-1.,0.,1.,0.]).to(device))
# how to see the model? -> winning accuracy of the player who played the last move in the input grid
a.item()

# for the Tttcomputer class
a = model(torch.tensor([1.,0.,-1.,0.,1.,-1.,0.,1.,0.]).to(device)).exp()
probs = a/a.sum()
torch.multinomial(probs, 1)
# torch.multinomial(a, 1)
#  1 0 -1
#  0 1 -1
#  0 1 0

# play with the computer
class Play:
  def __init__(self):
    self.grid = np.zeros((3,3), dtype=np.int8) #3x3 grid will be implemented as a tensor in the shape of (3,3)
    self.current = False
    self.grid_list = torch.zeros((3,3), dtype=torch.float).view(-1).to(device)
    self.dec = 0
    self.turn = 1

  def computer(self):
    a = model(self.grid_list).exp()
    probs = a/a.sum()
    compute = torch.multinomial(probs, 1)
    print(f'Computer chose {compute}')
    self.dec = compute

  def round(self, user_input): #user_input is going to be 0 through 8
    x, y = user_input//3, user_input%3

    if self.grid[x, y] != 0:
      print('invalid placement')
      if self.turn == 1:
        a = int(input(f"Your turn. Pick a number from 0 to 8 "))
        self.round(a)
      else:
        pass
        # self.computer()
        # self.round(self.dec)

    self.grid[x, y] = self.turn
    print(self.grid)
    self.turn *= -1

    self.grid_list = torch.tensor(self.grid, dtype=torch.float).view(-1).to(device)
    print(self.grid_list)
    print(f'odds of winning:{model1(self.grid_list).item():.4f}')

  def decision(self): #to decide whether it's over or not, returns boolean value or None if it's a draw (for the 1st option)
    if 0 not in self.grid:
      self.current = True
      return

    if 3 in [*np.abs(self.grid.sum(0)).tolist(), *np.abs(self.grid.sum(1)).tolist()]:
      self.current = True
    elif np.abs(self.grid.trace()) == 3:
      self.current = True
    elif np.abs(self.grid[0,2] + self.grid[1,1] + self.grid[2,0]) == 3:
      self.current = True
    else:
      self.current = False

  def __call__(self):
    coin = int(input('wanna play first? click 1. wanna play second? click 0: '))
    if coin == 1:
      while True:
        user_input = int(input(f"Your turn. Pick a number from 0 to 8 "))
        self.round(user_input)
        self.decision()
        if self.current:
          print('Congrats you won!')
          break
        self.computer()
        self.round(self.dec)
        self.decision()
        if self.current:
          print('Sorry you lost...')
          break
    else:
      while True:
        self.computer()
        self.round(self.dec)
        self.decision()
        if self.current:
          print('Sorry you lost...')
          break
        user_input = int(input(f"Your turn. Pick a number from 0 to 8 "))
        self.round(user_input)
        self.decision()
        if self.current:
          print('Congrats you won!')
          break
game = Play()
game()

#stuff to improve on
#1. Decision function is too messy
#2. Datamaker class can be optimized
#3. What to do when it's a draw (1st option: end it as a draw(hard to inplement in reinforcement learning), (returns None when the game is a draw)
# 2nd option: end it as B losing(it would be B's turn and the model could hack the game by just keep on drawing))
# -> trying the 2nd option
#4. What to do when you play in the same place (1st option: just end the game with the player losing, 2nd option: retry) -> tring the 2nd option
# the 2nd option may end as an infinite loop when playing with the reinforcement learning model since it just keeps retrying so probably should add count and add it to loss function
#5. A nearly infinite loop occurs when the computer spits out the same play

#takeaways
#1. Sometimes I build stuff that I don't even understand and it works so I have to go back and find out what it does lol
#2. The deeper the models gets, the more I forget what I wanted to implement or what certain functions do so always write down comments
#3. It is way better to first plan out what functions do what task with what i/o than just brute making stuff because you'll realize at some point what you're building is wrong
#(the Tttprediction class was an accident. I was only trying to build the Tttcomputer)
#4. Surprizingly enough, there are no fluctuation in the loss graph. I assume it's because the loss_interval is high plus the Adam optimizer.
#Or maybe it's just a RL thing. I should learn more about it.
#5. I wanted to add weights to games that take too many rounds (give higher weight to those rounds that ended earlier)
#It was such a mess. I basically had to dissect the crossentropyloss to multiply the weights. Also, nested for loops is never a good idea...
#6. So the most important thing I've learned here is that the RL model I've made is very limited to certain models and is not applicable to other models
#therefore I need to learn a more general RL model
