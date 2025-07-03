from tictactoe import TicTacToe
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

class Net(nn.Module):
    def __init__(self, x, h1, h2, y):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, y),
        )

    def forward(self, x):
        return self.net(x)

def the_model(observations):
    stats = torch.load('stats.pt')
    train_mean, train_std = stats['mean'], stats['std']

def how(game, player):
    board = game.board[:]
    board.append(player)
    board = tuple(board)
    move = 11 #move = the_model(board)
    possible = game.possible()
    if move in possible:
        return ({board : move}, move)
    else:
        move = random.choice(possible)
        return ({board : move}, move)

def play(game):
    data = []
    # A list containing a dictionary: {observations: action},
    # where observations = [game.board, player]
    # and action = move (int)
    player = game.judge()
    if player == "X":
        other = "O"
    else:
        other = "X"

    while game.end() == False:
        if game.turn == player:
            it = how(game, player)
            data.append(it[0])
            position = it[1]
            game.move(player, position)
        else:
            game.rand_move(other)
    if game.result() == player: # Should ties be included?
        return data
    return 0

# data is a list of dictionaries: {observations : actions}
def preprocess(data):
    inputs = []
    targets = []
    for d in data:
        x = list(d.keys())[0]
        float_x = []
        for x_v in x:
            if x_v == "X":
                float_x.append(-2.0)
            elif x_v == "O":
                float_x.append(-1.0)
            else:
                float_x.append(float(x_v))
        inputs.append(float_x)
        targets.append(list(d.values())[0])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    dataset = TensorDataset(inputs, targets)
    train_mean = train_inputs.mean(dim=0)
    train_std = train_inputs.std(dim=0)
    torch.save({'mean': train_mean, 'std': train_std}, 'stats.pt')
    return dataset

def process_inputs(data):
    inputs = []
    for d in data:
        x = list(d.keys())[0]
        float_x = []
        for x_v in x:
            if x_v == "X":
                float_x.append(-2.0)
            elif x_v == "O":
                float_x.append(-1.0)
            else:
                float_x.append(float(x_v))
        inputs.append(float_x)
        targets.append(list(d.values())[0])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def train(dataset):
    # Params
    x_dim = 10
    y_dim = 9
    h1 = 256
    h2 = 128
    batch_size = 64
    epochs = 200
    lr = 0.001
    ratio = 0.8
    train_size = int(ratio * len(dataset))

    torch.manual_seed(12)
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset)-train_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = 'cpu'
    model = Net(x_dim, h1, h2, y_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                a, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100 * correct/total
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), 'model.pt')

FIRST = []
for i in range(100000):
    game = TicTacToe()
    r = play(game)
    if r != 0:
        for a in r:
            FIRST.append(a)
print(len(FIRST))
dataset = preprocess(FIRST)
train(dataset)
