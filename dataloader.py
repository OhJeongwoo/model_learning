import torch
import torch.utils.data as data
import json

MAX_LEN = 100

class synthetic_example(data.Dataset):
    def __init__(self,path = './synthetic_example/data/sdata.json',train=True,split=0.1):
        
        self.path = path
        with open(self.path,'r') as jf:
            data = json.load(jf)
        self.x = torch.zeros(size=(0,2),dtype=torch.float64)
        self.y = torch.zeros(size=(0,2),dtype=torch.float64)
        self.noise = []
        self.vel = []
        self.type = []
        # Load Data
        for key in data:
            states = torch.FloatTensor(data[key]['states']).T
            actions = torch.FloatTensor(data[key]['actions']).T
            size = states.size()[0]
            save_file = key.zfill(6) + ".json"
            rt = {}
            rt['state'] = states.tolist()
            rt['action'] = actions.tolist()
            rt['length'] = size
            with open(save_file, 'w') as outfile:
                json.dump(rt, outfile, indent=4)
            vel = [data[key]['vel']]*size
            self.vel.extend(vel)
            noise = [data[key]['noise']]*size
            self.noise.extend(noise)
            type = [data[key]['type']]*size
            self.type.extend(type)
            self.x = torch.cat((self.x,states),dim=0)
            self.y = torch.cat((self.y,actions),dim=0)
        s = self.x.size()[0]
        self.noise = torch.FloatTensor(self.noise).unsqueeze(-1)
        self.vel = torch.FloatTensor(self.vel).unsqueeze(-1)
        self.type = torch.FloatTensor(self.type).unsqueeze(-1)
        # print(self.noise.size(),self.vel.size(),self.type.size())
        
        # Shuffle
        rand_idx = torch.randperm(s)
        self.x = self.x[rand_idx]
        self.y = self.y[rand_idx]
        self.noise = self.noise[rand_idx]
        self.vel = self.vel[rand_idx]
        self.type = self.type[rand_idx]

        # Train Test Split
        if not train:
            self.x = self.x[:int(s*split)]
            self.y = self.y[:int(s*split)]
            print(self.x.size(),self.y.size())
        else:
            self.x = self.x[int(s*split):]
            self.y = self.y[int(s*split):]
            print(self.x.size(),self.y.size())
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x,y
    
    def __len__(self):
        return self.x.size()[0]

if __name__ == '__main__':
    temp = synthetic_example()
    print(temp.__len__())