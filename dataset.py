import math
import torch
import torch.utils.data as Data
import numpy as np
import scipy.io as sio
import copy

class Dataset_train(Data.Dataset):
    def __init__(self, opt):
        super(Dataset_train, self).__init__()
        
        self.opt = opt
        
        data_dir = './data/'
        self.image_file = data_dir + opt.dataset + '.mat'
        
        self.input_data = sio.loadmat(self.image_file)
        self.image = self.input_data['data']
        self.image = self.image.astype(np.float32)
        
        self.col = self.image.shape[0]
        self.row = self.image.shape[1]
        self.band = self.image.shape[2]

        self.mirror_image = get_mirror_hsi(self.row, self.col, self.band, self.image, self.opt.patch)
        
        self.train_point = []
        self.train_data = np.zeros((self.col * self.row, self.opt.patch, self.opt.patch, self.band), dtype=float)
        self.size_data = (self.opt.patch, self.opt.patch, self.band)
        for i in range(self.row):
            for j in range(self.col):
                self.train_point.append([i,j])
        for k in range(len(self.train_point)):
            self.train_data[k,:,:,:] = get_neighborhood_pixel(self.mirror_image, self.train_point, k, self.opt.patch)
        self.len = self.train_data.shape[0]
         
        self.label = self.train_data
        self.input, self.mask = generate_mask(self.opt.ratio, self.opt.size_window, self.size_data, copy.deepcopy(self.label))

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        
        label = self.label[index]
        input = self.input[index]
        mask = self.mask[index]
        
        data = {'label': label, 'input': input, 'mask': mask}
        
        data = ToTensor(data)
        
        return data


def BS3LNetData(opt):
    
    # train dataloader
    dataset_train = Dataset_train(opt)
    loader_train = Data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
    
    data_dir = './data/'
    image_file = data_dir + opt.dataset + '.mat'
    
    input_data = sio.loadmat(image_file)
    image = input_data['data']
    band = image.shape[2]
    
    print("The construction process of training patch pairs with blind-spots is done")
    print('-' * 50)
    
    return loader_train, band


def generate_mask(ratio, size_window, size_data, input):
    
    #input row*col, patch, patch, band
    #size_data patch patch band
    #size_window candidate window size
    
    num_sample = math.ceil(size_data[0] * size_data[1] * (1 - ratio))
    mask = np.ones(input.shape)
    output = input
    
    for num in range(input.shape[0]):
        
        output_squeeze = output[num,:].squeeze()
        input_squeeze = input[num,:].squeeze()
        mask_squeeze = mask[num,:].squeeze()
        
        # center pixel
        exclude = (size_data[0] // 2, size_data[1] // 2)
        
        idy_msks = [exclude[0]]
        idx_msks = [exclude[1]]
        
        for ich1 in range(num_sample-1):
            idy_msk, idx_msk = exclude_random(exclude, size_data[0], size_data[1], center=False)
            idy_msks = np.append(idy_msks, idy_msk)
            idx_msks = np.append(idx_msks, idx_msk)
        
        # center point
        exclude_window = (0, 0)
        
        idy_neighs = []
        idx_neighs = []
        
        for ich2 in range(num_sample):
            idy_neigh, idx_neigh = exclude_random(exclude_window, size_window, size_window, center=True)
            idy_neighs.append(idy_neigh)
            idx_neighs.append(idx_neigh)
        
        idy_msk_neigh = idy_msks + idy_neighs
        idx_msk_neigh = idx_msks + idx_neighs
        
        idy_msk_neighs = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
        idx_msk_neighs = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]
        
        for ich in range(size_data[2]):
            id_msk = (idy_msks, idx_msks, ich)
            id_msk_neigh = (idy_msk_neighs, idx_msk_neighs, ich)
            
            output_squeeze[id_msk] = input_squeeze[id_msk_neigh]
            mask_squeeze[id_msk] = 0.0
        
        output[num,:] = np.expand_dims(output_squeeze, axis=0)
        mask[num,:] = np.expand_dims(mask_squeeze, axis=0)
        
    return output, mask


def exclude_random(exclude, y_size, x_size, center=False):
    
    exclude = exclude
    if center == False:
        idy_msk = np.random.randint(0, y_size)
        idx_msk = np.random.randint(0, x_size)
    else:
        idy_msk = np.random.randint(-y_size // 2 + y_size % 2, y_size // 2 + y_size % 2)
        idx_msk = np.random.randint(-x_size // 2 + x_size % 2, x_size // 2 + x_size % 2)
        
    if exclude[0]==idy_msk and exclude[1]==idx_msk:
        return exclude_random(exclude, y_size, x_size, center)
    else:
        return idy_msk, idx_msk
    
    
def ToTensor(data):
    """Convert ndarrays in sample to Tensors."""
        # Swap color axis because numpy image: N x H x W x C
        #                         torch image: N x C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((0, 3, 1, 2)))
        # return data

    input, label, mask = data['input'], data['label'], data['mask']

    input = input.transpose((2, 0, 1)).astype(np.float32)
    label = label.transpose((2, 0, 1)).astype(np.float32)
    mask = mask.transpose((2, 0, 1)).astype(np.float32)
    
    return {'input': torch.from_numpy(input),
            'label': torch.from_numpy(label),
            'mask': torch.from_numpy(mask)}


def get_mirror_hsi(height, width, band, image, patch):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band), dtype=float)
    #central region
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=image
    #left region
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=image[:,padding-i,:]
    #right region
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=image[:,width-2-i,:]
    #top region
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i,:,:]
    #bottom region
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-2-i,:,:]

    print('-' * 50)
    print("The patch size is : [{0},{1}]".format(patch,patch))
    print("The mirror_data size : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print('-' * 50)
    return mirror_hsi


def get_neighborhood_pixel(mirror_image, train_point, i, patch):
    x = train_point[i][0]
    y = train_point[i][1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image