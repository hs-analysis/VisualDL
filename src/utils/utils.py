from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import yaml

def train_all_epochs(model, train_loader, valid_loader, test_loader, epochs, criterions, criterion_scaling = None, average_outputs = False):
    
    if criterion_scaling is None:
        criterion_scaling = [1] * len(criterions)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler(enabled = False if device == 'cpu' else True)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    model = model.to(device)
    
    for epoch in range(epochs):
        training_bar = tqdm(train_loader)
        train_one_epoch(model, training_bar, criterions, criterion_scaling, average_outputs, device, epoch, optimizer, scaler)
        



def train_one_epoch(model, training_bar, criterions, criterion_scaling, average_outputs = False, device = None, epoch = 0, optimizer = None, scaler = None): 
    for cnt, (x,y) in enumerate(training_bar):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        #TODO implement average_outputs
        with torch.cuda.amp.autocast():
            loss = None
            predictions = model(x)
            for cr, scal in zip(criterions, criterion_scaling):
                if loss is None:
                    loss = cr(predictions, y) / scal
                else:
                    loss += cr(predictions, y) / scal
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_bar.set_description("Train: Epoch:%i, Loss:%.4f" % (epoch, loss.item()))       
                
                
    
def evaluate(loader, criterions, criterion_scaling, average_outputs = False):
    pass
    
    
def get_dataloader(dataset, batch_size, workers, shuffle = True):
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = workers)
    

def parse_yaml(yaml_file):
    with open(yaml_file, "r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)



    
    
    
    
