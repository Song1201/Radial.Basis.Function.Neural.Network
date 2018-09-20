
# coding: utf-8

# In[ ]:

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.stats as ss
import sys


# In[ ]:

def train(learning_rate,wd,iteration,x_tra, y_tra, Net, name):
    
    # Set up some basic parameters.    
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    # Convert numpy array to torch.Tensor 
    x_tra_t = torch.FloatTensor(x_tra)
    y_tra_t = torch.FloatTensor(y_tra)

    # Normalization
    #x_tra_mean_t = torch.mean(x_tra_t,0).view(1,3)
    #x_tra_std_t = torch.std(x_tra_t,0).view(1,3)
    #x_tra_t = (x_tra_t - x_tra_mean_t.repeat(x_tra_t.size(0),1))/x_tra_std_t.repeat(x_tra_t.size(0),1)

    # Convert Tensor to Variable
    x_tra_v = Variable(x_tra_t)
    y_tra_v = Variable(y_tra_t, requires_grad = False)
    #print('Data has been arranged!')

    net = Net()
    x_tra_vc = x_tra_v.cuda()
    y_tra_vc = y_tra_v.cuda()
    net.cuda()

    y_tra_moved_vc, del_tra_vc, sca_tra_vc = net(x_tra_vc, y_tra_vc)  

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate, betas=(beta_1,beta_2),eps=epsilon, weight_decay = wd)

    y_tra_mean_v = torch.mean(y_tra_v,0).view(1,768).repeat(x_tra_t.size(0),1)
    y_tra_mean_vc = y_tra_mean_v.cuda()
    loss_all = np.empty([iteration], dtype = np.float64)
    
    for step in range(iteration):
        optimizer.zero_grad()
        y_tra_moved_vc, del_tra_vc, sca_tra_vc = net(x_tra_vc, y_tra_vc)
        loss_2 = criterion(y_tra_moved_vc, y_tra_mean_vc)
        loss_all[step] = loss_2.data.cpu().numpy()
        loss_2.backward()
        optimizer.step()

    loss_tra = loss_2.data.cpu().numpy()
    print('loss_tra:')        
    print(loss_tra)
    
    x = np.arange(iteration)
    plt.figure(figsize = [15,10])
    plt.plot(x,loss_all, color = 'b')
    plt.show()
    

    torch.save(net.state_dict(), name)


# In[ ]:

def define_model(kernel,nar,kernel_s,nar_s, del_max, sca_max):
    # nar/ NAR is narrow, describe how narrow the kernel is.
    # Calculating phi
    def make_rbf_matrix(n_rbf, N, NAR = 0.02):
        def distance(x1, x2):
            d = float(abs(x1 - x2))
            if d > N // 2:
                d = N - d
            return d
        if(n_rbf > 1):
            interval = N // (n_rbf - 1)
            kernel_vector = [i * interval for i in range(n_rbf - 1)] + [400]
            NAR_vector = [NAR for i in range(n_rbf - 1)] + [0]
        else:
            kernel_vector = [600]
            NAR_vector = [0]
        rbf_matrix = np.array([[np.exp(-(distance(i, (kernel_vector[j]) % N) * NAR_vector[j])**2)
                                for j in range(n_rbf)] for i in range(N)], dtype=np.float32)
        return rbf_matrix

    phi_t = torch.from_numpy(make_rbf_matrix(kernel, 768, NAR = nar))
    phi_t = torch.t(phi_t)
    phi_v = Variable(phi_t, requires_grad = True)
    phi_vc = phi_v.cuda()

    # Calculating phi for scaling.
    phi_s_t = torch.from_numpy(make_rbf_matrix(kernel_s, 768, NAR = nar_s))
    phi_s_t = torch.t(phi_s_t)
    phi_s_v = Variable(phi_s_t, requires_grad = True)
    phi_s_vc = phi_s_v.cuda()

    class Horizotal_Move(Function):

        def forward(self, delta, y):
            self.save_for_backward(delta, y)
            # position of every point after moving
            new_pos = delta + torch.arange(0, 768).view(1,768).expand(delta.size(0),768).cuda() 
            new_pos_fore = (torch.floor(new_pos) % 768).long()
            new_pos_back = (new_pos_fore+1) % 768 # right int point of the moved position
            y_moved_back = torch.gather(y, 1, new_pos_back)
            y_moved_fore = torch.gather(y, 1, new_pos_fore)
            y_moved = y_moved_fore + (new_pos-torch.floor(new_pos))*(y_moved_back-y_moved_fore)
            return y_moved

        def backward(self, grad_output):
            delta, y = self.saved_tensors
            # position of every point after moving
            new_pos = delta + torch.arange(0, 768).view(1,768).expand(delta.size(0),768).cuda() 
            new_pos_fore = (torch.floor(new_pos) % 768).long()
            new_pos_back = (new_pos_fore+1) %768
            y_moved_back = torch.gather(y, 1, new_pos_back)
            y_moved_fore = torch.gather(y, 1, new_pos_fore)
            grad_delta = grad_output * (y_moved_back - y_moved_fore)
            return grad_delta, None

            #return grad_input, grad_weight, grad_bias

    def horizotal_move(delta,y):
        return Horizotal_Move()(delta, y)

    #print('horizotal_move has been defined!')
    

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__() # Inherit Net's fathers initiation
            self.fc1 = nn.Linear(3, kernel, bias=False) 
            init.uniform(self.fc1.weight, 0, 1)
            self.fc2 = nn.Linear(3, kernel_s, bias = False)
            init.uniform(self.fc1.weight, 0, 1)
 
        def forward(self, x, y):
            x_m = self.fc1(x)
            delta = torch.mm(x_m, phi_vc) # the distance each points needs to be moved
            delta = del_max * torch.tanh(delta)
            y_moved = horizotal_move(delta,y)
            x_s = self.fc2(x)
            scale = torch.mm(x_s, phi_s_vc)
            scale = sca_max * torch.tanh(scale)
            scale = 1 + scale
            y_scale = y_moved*scale
            return y_scale, delta, scale
        
    return Net


# In[ ]:

def denoise(net, x_valid, y_valid):
    # Convert numpy array to torch.Tensor 
    x_valid_t = torch.FloatTensor(x_valid)
    y_valid_t = torch.FloatTensor(y_valid)

    # Convert Tensor to Variable
    x_valid_v = Variable(x_valid_t)
    y_valid_v = Variable(y_valid_t)

    # Move data to GPU
    x_valid_vc = x_valid_v.cuda()
    y_valid_vc = y_valid_v.cuda()

    y_valid_moved_vc, delta_vc, scale_vc = net(x_valid_vc, y_valid_vc)
    y_valid_moved = y_valid_moved_vc.data.cpu().numpy()
    delta = delta_vc.data.cpu().numpy()
    scale = scale_vc.data.cpu().numpy()
    
    return y_valid_moved, delta, scale


# In[ ]:

def diagnose(t, t_mean, t_std, c, u):
    # t_mean is a numpy 1 dim array 
    t_tc = torch.FloatTensor(t).cuda()
    m, n = t_tc.size() # The height and nar
    t_bou = ss.norm(t_mean, t_std).ppf(c) # The decision boundary of t
    #print(t_bou)
    t_bou_tc = torch.FloatTensor(t_bou).cuda()
    t_bou_tc = t_bou_tc.repeat(m,1)
    exc_tc = t_tc < t_bou_tc # exc is exceed
    exc_tc.resize_(m,n,1)
    for i in range(1,u):
        exc_temp_tc = torch.cat((exc_tc[:,i:,0],exc_tc[:,0:i,0]),1)
        exc_tc = torch.cat((exc_tc,exc_temp_tc),2)

    result = torch.cumprod(exc_tc,2)
    result = result[:,:,u-1]
    result = result.short()
    result = result.sum(1)
    result = result > 0
    result = result.cpu().numpy()
    return result
    


# In[ ]:

def calculate_false_true(result, label):
    false_positive = (np.sum(result)-np.sum(result*label))/(label.shape[0]-np.sum(label))
    true_positive = np.sum(result*label)/np.sum(label)
    return false_positive, true_positive


# In[ ]:

def calculate_roc_con(t, t_mean, t_std, label, conf_range, under_num):
    #con is confidence
    x = np.empty([0], dtype = np.float64)
    y = np.empty([0], dtype = np.float64)
    for c in conf_range:
        result = diagnose(t, t_mean, t_std, c, under_num)
        false_positive, true_positive = calculate_false_true(result, label)
        x = np.append(x, false_positive)
        y = np.append(y, true_positive)
    #y = y[np.argsort(x)]
    #x = np.sort(x)

    return x, y


# In[ ]:

def calculate_roc_und(t, t_mean, t_std, label, con, und_ran):
    #under_number
    x = np.empty([0], dtype = np.float64)
    y = np.empty([0], dtype = np.float64)
    for u in und_ran:
        result = diagnose(t, t_mean, t_std, con, u)
        false_positive, true_positive = calculate_false_true(result, label)
        x = np.append(x, false_positive)
        y = np.append(y, true_positive)
    #y = y[np.argsort(x)]
    #x = np.sort(x)

    return x, y


# In[ ]:

def draw_roc_con(x1, y1, x2, y2, under_num, x_ran):
    # x_ran is x_range  con is confidence
    plt.figure(figsize=(4.5*x_ran,4.5)) 
    plt.plot(np.extract(x1<=x_ran, x1), np.extract(x1<=x_ran, y1),color = 'r') 
    plt.plot(np.extract(x2<=x_ran, x2), np.extract(x2<=x_ran, y2),color = 'b')
    plt.axis([-0.01, x_ran, -0.01, 1])
    plt.title('fixing under_num={0}'.format(under_num))
    plt.show()


# In[ ]:

def draw_roc_und(x1, y1, x2, y2, con, x_ran):
    # x_ran is x_range  con is confidence
    plt.figure(figsize=(4.5*x_ran,4.5)) 
    plt.plot(np.extract(x1<=x_ran, x1), np.extract(x1<=x_ran, y1),color = 'r') 
    plt.plot(np.extract(x2<=x_ran, x2), np.extract(x2<=x_ran, y2),color = 'b')
    plt.axis([-0.01, x_ran, -0.01, 1])
    plt.title('fixing conf={0}'.format(con))
    plt.show()


# In[ ]:

def calculate_auc(x, y):
    # In this RNFLT case I don't resort the data, because the data I acquired is already in order. But in more common cases, 
    # the data should be reordered before caculating the auc.
    x_r_shi = np.append(1, x[0:x.shape[0]-1]) # x_right_shift
    del_x = np.absolute(x_r_shi - x)
    auc = np.inner(del_x, y)
    return auc
    


# In[ ]:

def cross_shift(con, ill, i, who):
    # con is control, who is whole
    # gro is group
    gro = int(con.shape[0]/who)   
    val = np.concatenate((con[i*gro:(i+1)*gro],ill),0)
    np.random.shuffle(val)
    tra = np.concatenate((con[0:i*gro], con[(i+1)*gro:]),0)
    np.random.shuffle(tra)
    return tra, val


def divideData(control,ill,groupNo):
    # groupNo should be less than the maximum group number, in this case is 9. 
    maxGroupNo = int(control.shape[0]/ill.shape[0])
    if groupNo > maxGroupNo:
        sys.exit('groupNo>maxGroupNo')
    
    valid = np.concatenate((control[(groupNo-1)*ill.shape[0]:groupNo*ill.shape[0]],ill),axis=0)
    train = np.concatenate((control[0:(groupNo-1)*ill.shape[0]],control[groupNo*ill.shape[0]:]),axis=0)
    
    xValid = valid[:,0:3]
    yValid = valid[:,3:771]
    
        
    return valid, train