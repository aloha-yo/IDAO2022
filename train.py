from gatgnn.data                   import *
from gatgnn.model                  import *
from gatgnn.pytorch_early_stopping import *
from gatgnn.file_setter            import use_property
from gatgnn.utils                  import *
import os

crystal_property = 'new-property'
data_src = 'NEW'
RSM   = {'radius':8,'step':0.2,'max_num_nbr':12}
norm_action, classification           = set_model_properties(crystal_property)
training_num = 0.75
number_layers = 5
number_neurons = 64
n_heads = 4
xtra_l = True
global_att = 'composition'
attention_technique = 'random'
concat_comp = False

# SETTING UP CODE TO RUN ON GPU
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# DATA PARAMETERS
random_num  =  228;
random.seed(random_num)

# MODEL HYPER-PARAMETERS
num_epochs      = 400
learning_rate   = 5e-3
batch_size      = 64

stop_patience   = 150
best_epoch      = 1
adj_epochs      = 200
milestones      = [100,150,200,250,300,350]
train_param     = {'batch_size':batch_size, 'shuffle': True}
valid_param     = {'batch_size':batch_size, 'shuffle': True}

# DATALOADER/ TARGET NORMALIZATION
src_CIF         = 'CIF-DATA_NEW'
dataset         = pd.read_csv(f'data/dichalcogenides_public/targets.csv').sample(frac=1,random_state=random_num)
dataset.columns = ['material_ids','label']

NORMALIZER      = DATA_normalizer(dataset.label.values)

CRYSTAL_DATA    = CIF_Dataset(dataset, root_dir = f'data/dichalcogenides_public/structures/',**RSM)
idx_list        = list(range(len(dataset)))
random.shuffle(idx_list)

train_idx, val_idx = train_test_split(idx_list,train_size=training_num,random_state=random_num)

training_set       =  CIF_Lister(train_idx,CRYSTAL_DATA,NORMALIZER,norm_action,df=dataset,src=data_src)
validation_set     =  CIF_Lister(val_idx,CRYSTAL_DATA,NORMALIZER,norm_action,  df=dataset,src=data_src)

# NEURAL-NETWORK
the_network    = GATGNN(n_heads,classification,neurons=number_neurons,nl=number_layers,xtra_layers=xtra_l,global_attention=global_att,
                                      unpooling_technique=attention_technique,concat_comp=concat_comp,edge_format=data_src)
net            = the_network.to(device)
#net.load_state_dict(torch.load(f'TRAINED/best.pt',map_location=device))

# LOSS & OPTMIZER & SCHEDULER
criterion   = nn.SmoothL1Loss(beta=1).cuda()
funct = torch_MAE
optimizer = optim.Adam(net.parameters(), lr = learning_rate)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.3)
metrics = METRICS(crystal_property,num_epochs,criterion,funct,device)


train_loader   = torch_DataLoader(dataset=training_set,   **train_param)
valid_loader   = torch_DataLoader(dataset=validation_set, **valid_param) 

for epoch in range(num_epochs):
    net.train()       
    start_time       = time.time()
    for i, data in enumerate(train_loader):
        
        data         = data.to(device)
        predictions  = net(data)
        train_label  = metrics.set_label('training',data)
        loss         = metrics('training',predictions,train_label,1)
        _            = metrics('training',predictions,train_label,2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics.training_counter+=1
    metrics.reset_parameters('training',epoch)
    # VALIDATION-PHASE
    net.eval()
    for i, data in enumerate(valid_loader):
        
        data = data.to(device)
        with torch.no_grad():
            predictions    = net(data)
        valid_label        = metrics.set_label('validation',data)
        _                  = metrics('validation',predictions,valid_label,1)
        _                  = metrics('validation',predictions, valid_label,2)

        metrics.valid_counter+=1

    metrics.reset_parameters('validation',epoch)
    scheduler.step()
    end_time         = time.time()
    e_time           = end_time-start_time
    metrics.save_time(e_time)
    estop_val        = '@best: saving model...'; best_epoch = epoch+1
    torch.save(net.state_dict(), f'TRAINED/crystal-checkpoint{epoch}.pt')


