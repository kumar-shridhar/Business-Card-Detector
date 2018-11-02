############### Configuration file ###############
import math

start_epoch = 1
num_epochs = 100
batch_size = 32
optim_type = 'Adam'

mean = {
    'business_cards': (0.485, 0.456, 0.406),
}

std = {
    'business_cards': (0.229, 0.224, 0.225),
}


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
