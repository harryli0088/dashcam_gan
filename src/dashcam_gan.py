import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from get_latest_model import get_latest_model
from azureml.core import Run


# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to the training data', default="./data/image_data_train")
# parser.add_argument('--dashcam_model', type=str, help='Path to the dashcam gan models', default="./dashcam_model/")
# parser.add_argument('--dashcam_samples', type=str, help='Path to the dashcam generator samples', default="./dashcam_samples")
args = parser.parse_args()
run = Run.get_context()



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# some hyperparameters
BATCH_SIZE = 1 #00 # 1
NOISE_DIM = 100
EPOCHS = 200
LEARNING_RATE = 0.0002
loss_function = nn.BCELoss()


# load the data
# Azure can store files in the outputs/ directory
TRAIN_DATA_PATH = args.data_path
DASHCAM_MODEL_PATH = "./outputs/" # args.dashcam_model
DASHCAM_SAMPLE_PATH = "./outputs/"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)




# define the Generator and Discriminator

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


# build Generator and Discriminator
dashcam_dim = train_dataset[0][0].shape[0] * train_dataset[0][0].shape[1] * train_dataset[0][0].shape[2]
print(train_dataset[0][0].shape[0],train_dataset[0][0].shape[1],train_dataset[0][0].shape[2])
print("dashcam_dim",dashcam_dim)
G = Generator(g_input_dim = NOISE_DIM, g_output_dim = dashcam_dim).to(device)
D = Discriminator(dashcam_dim).to(device)

# used saved models if you have them
saved_G = get_latest_model("g-",DASHCAM_MODEL_PATH)
saved_D = get_latest_model("d-",DASHCAM_MODEL_PATH)
if len(saved_G["filepath"]) > 0:
    G.load_state_dict(torch.load(saved_G["filepath"]))
    D.load_state_dict(torch.load(saved_D["filepath"]))
    EPOCHS = EPOCHS - saved_G["latest_epoch"]
print("EPOCHS",EPOCHS)

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr = LEARNING_RATE)
D_optimizer = optim.Adam(D.parameters(), lr = LEARNING_RATE)


def generate_images(file_path=""):
    with torch.no_grad():
        test_z = Variable(torch.randn(BATCH_SIZE, NOISE_DIM).to(device))
        generated = G(test_z) # generate images

        save_image(generated.view(generated.size(0), 3, 256, 256),file_path )


def D_train(x):
    #=======================Train the discriminator=======================#

    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, dashcam_dim), torch.ones(BATCH_SIZE, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = loss_function(D_output, y_real)
    D_real_score = D_output


    # train discriminator on fake
    z = Variable(torch.randn(BATCH_SIZE, NOISE_DIM).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(BATCH_SIZE, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = loss_function(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()

def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(BATCH_SIZE, NOISE_DIM).to(device)) #get noise input
    y = Variable(torch.ones(BATCH_SIZE, 1).to(device)) #ones, ie G wants D to think this is real

    G_output = G(z) #give G input noise
    D_output = D(G_output) #pass the output from G into D
    G_loss = loss_function(D_output, y) #get the loss between the desired D output and the actual D output

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

# run epochs
for epoch in range(1, EPOCHS+1):
    D_losses, G_losses = [], []

    # run each batch in the data
    for batch_idx, (x, _) in enumerate(train_data_loader):
        print("epoch", epoch, "batch_idx", batch_idx, "x.size()[0]", x.size()[0])

        # sometimes the input batch size isn't big enough, I'm assuming because there is a remainder number of input samples
        # only train the D and G if the batch size is correct
        if x.size()[0]==BATCH_SIZE:
            D_losses.append(D_train(x))
            G_losses.append(G_train(x))
            break

    # after each epoch, save the state of each model
    filename_base = "-epoch-"+str(epoch)
    torch.save(G.state_dict(), DASHCAM_MODEL_PATH+"/g"+filename_base)
    torch.save(D.state_dict(), DASHCAM_MODEL_PATH+"/d"+filename_base)
    print('epoch', epoch)
    print('loss_d', torch.mean(torch.FloatTensor(D_losses)).item())
    print('loss_g', torch.mean(torch.FloatTensor(G_losses)).item())
    run.log('loss_d', torch.mean(torch.FloatTensor(D_losses)).item())
    run.log('loss_g', torch.mean(torch.FloatTensor(G_losses)).item())

    # save full batch of generated images
    generate_images(DASHCAM_SAMPLE_PATH+'/sample_' + str(epoch) + '.png')




# generate a final set of images
generate_images(DASHCAM_SAMPLE_PATH+'/sample_final' + '.png')
