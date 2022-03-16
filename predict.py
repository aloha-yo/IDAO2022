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
device = 'cpu'
# DATA PARAMETERS
random_num  =  228;
random.seed(random_num)


the_network    = GATGNN(n_heads,classification,neurons=number_neurons,nl=number_layers,xtra_layers=xtra_l,global_attention=global_att,
                                      unpooling_technique=attention_technique,concat_comp=concat_comp,edge_format=data_src)
net            = the_network.to(device)
net.load_state_dict(torch.load('TRAINED/checkpoint.pt', map_location='cpu'))



test_param      = {'batch_size': 2, 'shuffle': False}


dataset = pd.DataFrame([[i[:-5], 0.1] for i in os.listdir('data/dichalcogenides_private/structures/') if i != 'atom_init.json'], columns= ['material_ids', 'label'])

NORMALIZER              = DATA_normalizer(dataset.label.values)

CRYSTAL_DATA          = CIF_Dataset(dataset, root_dir = 'data/dichalcogenides_private/structures/',**RSM)

test_idx              = list(range(len(dataset)))
testing_set           = CIF_Lister(test_idx,CRYSTAL_DATA,NORMALIZER,norm_action, df=dataset,src=data_src)


test_loader = torch_DataLoader(dataset=testing_set,    **test_param)


net.eval()

pred = []

for i, data in enumerate(test_loader):
    if i%50 == 0:
        print(i)
    data  = data.to(device)
    with torch.no_grad():
        prediction = net(data)
    try:
        pred += prediction.cpu().tolist()
    except:
        pred.append(prediction.cpu())
dataset['label'] = pred
dataset.columns = ['id', 'predictions']
dataset.to_csv('submission.csv', header = True, index=False)

