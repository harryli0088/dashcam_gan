# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from get_latest_model import get_latest_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




BATCH_SIZE = 100
NOISE_DIM = 100
EPOCHS = 200
TRAIN_DATA_PATH = "./image_data_train"
TEST_DATA_PATH = "./image_data_test"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)



class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


# build network

dashcam_dim = train_dataset[0][0].shape[0] * train_dataset[0][0].shape[1] * train_dataset[0][0].shape[2]

saved_G = get_latest_model("g-","./dashcam_model/")
saved_D = get_latest_model("d-","./dashcam_model/")
G = Generator(g_input_dim = NOISE_DIM, g_output_dim = dashcam_dim).to(device)
D = Discriminator(dashcam_dim).to(device)

if len(saved_G["filepath"]) > 0:
    G.load_state_dict(torch.load(saved_G["filepath"]))
    D.load_state_dict(torch.load(saved_D["filepath"]))
    EPOCHS = EPOCHS - saved_G["latest_epoch"]


# loss
criterion = nn.BCELoss()

# optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)


def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, dashcam_dim), torch.ones(BATCH_SIZE, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(BATCH_SIZE, NOISE_DIM).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(BATCH_SIZE, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()

def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(BATCH_SIZE, NOISE_DIM).to(device))
    y = Variable(torch.ones(BATCH_SIZE, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

for epoch in range(1, EPOCHS+1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_data_loader):
        print("epoch", epoch, "batch_idx",batch_idx)
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    filename_base = "-epoch-"+str(epoch)
    torch.save(G.state_dict(), "./dashcam_model/g"+filename_base)
    torch.save(D.state_dict(), "./dashcam_model/d"+filename_base)
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), EPOCHS, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

    with torch.no_grad():
        test_z = Variable(torch.randn(BATCH_SIZE, NOISE_DIM).to(device))
        generated = G(test_z)

        save_image(generated.view(generated.size(0), 3, 256, 256), './dashcam_samples/sample_' + str(epoch) + '.png')


with torch.no_grad():
    test_z = Variable(torch.randn(BATCH_SIZE, NOISE_DIM).to(device))
    generated = G(test_z)

    save_image(generated.view(generated.size(0), 3, 256, 256), './dashcam_samples/sample_final' + '.png')
