import numpy as np
import time
import re
import os
import pandas as pd
import collections

import torch
import torch.cuda
import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt 
plt.style.use('seaborn-bright')
import matplotlib as mpl 
from cycler import cycler

#############################################################    
################## Visualization functions ##################
#############################################################
def visualization_traning_result (models):      
    #mpl.rc("figure", figsize=(10,8))
    fig, axes = plt.subplots(1,2,figsize=(16, 8), sharex='row')
   # plt.style.use('seaborn-bright')
    plt.rcParams["axes.prop_cycle"] = cycler('color', 
                        ['#332288', '#CC6677', '#DDCC77', '#117733', '#88CCEE', '#AA4499', 
                        '#44AA99', '#999933', '#882255', '#661100', '#6699CC', '#AA4466'])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    y_lim_min=10
    y_lim_max=0
    for model_id, color in zip(models.keys(),colors):
        #print(type(a_loss))
        if 'train_pred' in models[model_id]:
            x_train_pred = models[model_id]['train_pred']
            x_loss = models[model_id]['loss']
            axes[0].plot(x_train_pred.keys(), x_train_pred.values(), 
                    label=model_id,
                    color=color, 
                    linestyle='-', 
                    linewidth=3)
            axes[1].plot(x_loss.keys(), x_loss.values(), 
                    label=model_id,
                    color=color, 
                    linestyle='-', 
                    linewidth=3)
            if min(list(x_loss.values())) < y_lim_min:
                y_lim_min = min(list(x_loss.values()))
            if max(list(x_loss.values())) > y_lim_max:
                y_lim_max = max(list(x_loss.values()))
        axes[0].set_ylim(0.7,1)
        axes[1].set_ylim(y_lim_min,y_lim_max/2)

#    plt.xticks(fontsize = 18)
#    plt.yticks(fontsize = 18)
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[0].set_xlabel('Iter', fontsize = 18)
    axes[0].set_ylabel('Accuracy', fontsize = 18)
    axes[1].set_xlabel('Iter', fontsize = 18)
    axes[1].set_ylabel('Loss', fontsize = 18)
    fig.suptitle('How fit the model to your data?', fontsize=20)
    axes[0].legend(fontsize=12)
    axes[1].legend(fontsize=12)
    plt.show()


#### evaluation auc and precision and recall #######
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve
from scipy import interp   
def evaluation(test_set, model_result, note='', show=True):
    print('### Results:', note)
    print(time.strftime("%Y%m%d%H%M"))
    test_dum = pd.get_dummies(test_set)
    baseline = len(test_set[test_set==1])/(len(test_set[test_set==1])+len(test_set))
    print(baseline)

    auc, precision = get_accuracy(test_dum, model_result)
    save_result = note+time.strftime("%Y%m%d") +'_auc.png'
    if show==True:
        visualization_result(auc, precision, baseline, save_result, note)

 
def get_accuracy (test_set, model_result, n_classes=2):
    ### MACRO
    accuracy_performance = dict()
    accuracy_performance['auroc'] = collections.OrderedDict()
    accuracy_performance['precision'] = collections.OrderedDict()

    for model_name, item in model_result.items():
        accuracy_performance['auroc'][model_name] = {}# collections.OrderedDict()
        accuracy_performance['precision'][model_name] = {} #collections.OrderedDict()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        ### for precision-recall 
        pre = dict()
        rec = dict()
        thd = dict()
        pre_rec_score = dict()

        ''' micro '''
        ### calculate ROC curve for all classes 
        fpr["micro"], tpr["micro"], _ = roc_curve(test_set.values[:,1], item['y_pred_prob'][:,1].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        ### calculate precision-recall curve for all classes
        pre["micro"], rec["micro"], _ = precision_recall_curve(test_set.values[:,1], 
                                                item['y_pred_prob'][:,1].ravel())
        pre_rec_score["micro"]  = average_precision_score(test_set.values[:,1], 
                                                item['y_pred_prob'][:,1].ravel(), 'micro')
        

        ''' macro '''   
     #   from sklearn.preprocessing import OneHotEncoder
      #  tmp = np.array([[0],[1]])
      #  labeler = OneHotEncoder()
      #  labeler.fit(tmp)
       # test_set = labeler.transform(test_set.values.reshape(-1,1)).toarray()
      #  test_set = pd.DataFrame(test_set)
        for i in range(n_classes):
            ### calculate ROC curve for each class 
            fpr[i], tpr[i], _ = roc_curve(test_set.values[:, i], 
                                            item['y_pred_prob'][:, i],pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])    
            ### calculate precision-recall curve for each class --> does not calculate average_precision_score in macro 
            pre[i], rec[i], thd[i] = precision_recall_curve(test_set.values[:,i], item['y_pred_prob'][:, i],pos_label=i)
            pre_rec_score[i]  = average_precision_score(test_set.values[:, i], item['y_pred_prob'][:, i])
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = 0#all_fpr
        tpr["macro"] = 0#mean_tpr        
        roc_auc["macro"] = 0#auc(fpr["macro"], tpr["macro"])

        #accuracy_performance['auroc'][model_name]  = [fpr["macro"], tpr["macro"], roc_auc["macro"]]     
        accuracy_performance['auroc'][model_name]['fpr'] = fpr[1]
        accuracy_performance['auroc'][model_name]['tpr'] = tpr[1]
        accuracy_performance['auroc'][model_name]['roc_auc'] = roc_auc[1]

        if not os.path.exists('./log/'):
            os.makedirs('./log/')

        np.savetxt ('./log/' + model_name+time.strftime("%Y%m%d%H%M")+'.roc' , np.column_stack((fpr[1],tpr[1])), fmt='%.4f')

        accuracy_performance['precision'][model_name]['pre'] = pre[1]
        accuracy_performance['precision'][model_name]['rec'] = rec[1]
        accuracy_performance['precision'][model_name]['pre_rec_score'] = pre_rec_score[1]

        np.savetxt ('./log/' + model_name+time.strftime("%Y%m%d%H%M")+'.prec' , np.column_stack((rec[1],pre[1])), fmt='%.4f')
        #np.savetxt ('../log/' + model_name+'.thdprec' , thd[1], fmt='%.4f')
        #print((rec[1][-1]),(pre[1][-1]),(thd[1][-1]))


    #print(accuracy_performance['auroc'].items())
    auc_ranking = sorted(accuracy_performance['auroc'].items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    precision_ranking = sorted(accuracy_performance['precision'].items(), key=lambda x: x[1]['pre_rec_score'], reverse=True)

    print('AUC Ranking!')
    for i, model in enumerate (auc_ranking,1):
        print(i,model[0],'{:.3f}'.format(model[1]['roc_auc']))
    print('\n')

    print('Average Precision Ranking!')
    for i, model in enumerate (precision_ranking,1):
        print(i,model[0],'{:.3f}'.format(model[1]['pre_rec_score']))

    return accuracy_performance['auroc'], accuracy_performance['precision']


def visualization_result (auc, precision, baseline,file_name, note):
    plt.style.use('dark_background')   
    fig = plt.figure(figsize=(16,8))    
    ax1 = fig.add_subplot(1,2,1)
    
    plt.rcParams["axes.prop_cycle"] = cycler('color', 
                       [  '#CC6677','#DDCC77', '#117733', '#88CCEE', '#6699CC', '#44AA99', '#661100',  '#882255', '#999933','#AA4499',
                          '#AA4466','#332288',])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (key, item) , color in zip(auc.items(), colors):    
        ax1.plot(item['fpr'], item['tpr'], #i 번째 모델에 fpr 과 tpr
                 label=key+': {0:0.3f}'
                       ''.format(item['roc_auc']),
                 color=color, linestyle=':', linewidth=2)

    ax1.plot([0, 1], [0, 1], 'k',color='red', lw=1)
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_title(note + ' Cancer\nReceiver Operating Characteristic', fontsize=14)
    ax1.set_xlabel('1-Specificity', fontsize = 11)
    ax1.set_ylabel('Sensitivity', fontsize = 11)
    
    ax1.legend(fontsize=12)

    ax2 = fig.add_subplot(1,2,2)
    
    for (key, item) , color in zip(precision.items(), colors):    
        ax2.plot(item['rec'], item['pre'], 
                 label=key+': {0:0.3f}'
                       ''.format(item['pre_rec_score']),
                 color=color, linestyle=':', linewidth=2)


    ax2.plot([0, 1], [baseline, baseline], 'k',color='red' , lw=1) ## baseline
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(note + ' Cancer\nPrecision-Recall Curve', fontsize=14)
    ax2.set_xlabel('Recall', fontsize = 11)
    ax2.set_ylabel('Precision', fontsize = 11)
    ax2.legend(fontsize=11)

    ## save result
    fig.savefig(file_name ,dpi=300, transparent=True)
    plt.show()

########################################################################
############ Functions for deep learning ###############################
########################################################################
class torch_dataset (Dataset):
    def __init__(self,  x, y):
        self.x_data = torch.FloatTensor(x) 
        self.y_data = torch.FloatTensor(y) 
        self.len = len(x)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len

def create_variables_GPU(tensor):
    if torch.cuda.is_available():
       # print("Variables on Device memory")
        return Variable(tensor.cuda())
    else:
       # print("Variables on Host memory")
        return Variable(tensor)
    
def model_on_GPU (model):
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), 'GPUs used!')
        model = torch.nn.DataParallel(model).cuda()

    elif torch.cuda.is_available():
        print("Model on Device memory")
        model = model.cuda()
    else:
        print("Model on Host memory")

    return model         

class EarlyStopping():
    def __init__(self, patience=0, verbose=0, th = 0.001):
        self._step = 0
        self._zero = 0
        self._same = 0
        self.threshold = th
        self._loss = float('inf')
        self.patience  = patience
        self.verbose = verbose
 
    def validate(self, loss):
        #print(self._loss, loss)
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Training process is stopped early....')
                return True
        elif self._loss <self.threshold: # keep to reach zero 
            self._zero += 1
            if self._zero > self.patience:
                if self.verbose:
                    print('Training process is stopped early....')
                return True
        elif abs(self._loss - loss) < self.threshold:
            self._same += 1
            if self._same > self.patience:
                #print('samesame')
                if self.verbose:
                    print('Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._same = 0 
            self._zero = 0
            self._loss = loss
 
        return False


def cnt_pred(pred_prob, labels):
    return (len(pred_prob)-(abs(pred_prob -  labels)).sum())
    
def perform_deep (X_train, y_train, X_test, y_test, models, bat_size=32, epoch =10, lr =0.002 , CompCoef=False, note= ''): 
    model_result =collections.OrderedDict()

    print('- Input shape - ')
    print('Training:',X_train.shape, y_train.shape, '|| Test:',X_test.shape,y_test.shape)   

    ## data setting
    deep_train_set = DataLoader(dataset=torch_dataset(X_train,y_train),
        batch_size=bat_size, shuffle=True)
    deep_test_set = DataLoader(dataset=torch_dataset(X_test, y_test),
        batch_size=bat_size, shuffle=False)     
    
    for model in models:
        #early_stopping = EarlyStopping(patience=10, verbose=1) 
        loss_print = {} 
        train_acc_print = {}
        model_info ={}    
        ml_model_name = model._get_name()+note

        print('+'*50) 
        print('Model name:', ml_model_name)          
        model_info['name'] = ml_model_name
        model_info['module_info'] = 'LearnRate: '+ str(lr)+', Epoch: ' + str(epoch)

        model = model_on_GPU(model).train()      
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        t = time.process_time()      
        for it in range(epoch):
            epoch_loss = 0
            epoch_pred = 0
            for j, (x, y) in enumerate(deep_train_set, 1):
                inputs = create_variables_GPU(x.float())
                labels = create_variables_GPU(y.long())   
                if ml_model_name.find('Att')>-1:
                    output, _= model.forward(inputs)
                else:
                    output = model.forward(inputs)
                #loss = criterion(output, torch.max(labels,1)[1]) # torch.max -> indexes with 1 of 1 dimesion array 
                loss = criterion(output, labels) # torch.max -> indexes with 1 of 1 dimesion array 
                epoch_loss += loss.item()
               # epoch_pred += cnt_pred(torch.max (output,1)[1], torch.max(labels,1)[1]) #correct count
                epoch_pred += cnt_pred(torch.max (output,1)[1], labels) #correct count
                model.zero_grad()
                loss.backward()
                optimizer.step()
            l = epoch_loss
            loss_print.update({it:l})
            train_acc_print.update({it:int(epoch_pred.cpu())/len(deep_train_set)})
            ##if early_stopping.validate(l):
            #    break           
            if(it%10==0):
                print('{:,}th loss: {:.2f},'.format(it, l), end='  ')                    
        print(''.rjust(2),'{:,}th loss:{:.2f}'.format(it, l), 'Complete! Training time:', '{:.3f}'.format(time.process_time() - t),'sec')
        model_info['loss'] =loss_print
        model_info['train_pred'] =train_acc_print

        print('Testing...', end=' ') 
        att_weight=[]
        total_pred_prob =0
        t= time.process_time()
        model = model.eval()
        for it, (x, y) in enumerate(deep_test_set, 1):
            inputs = create_variables_GPU(x.float())
            labels = create_variables_GPU(y.long())  

            if ml_model_name.find('Att')>-1:
                y_pred_prob, weight= model.forward(inputs)
            else:
                y_pred_prob = model.forward(inputs)
            if ml_model_name.find('Att')>-1: #if not isinstance(weight,str):
                att_weight+= list(weight.cpu().detach().numpy())
        
            #total_pred += list(torch.max(y_pred_prob,1)[1].cpu().detach().numpy())
            if it ==1 :
                total_pred_prob=y_pred_prob.cpu().detach().numpy()
            else: 
                total_pred_prob = np.concatenate((total_pred_prob,y_pred_prob.cpu().detach().numpy()), axis=0)
        
        elapsed_time = time.process_time() - t
        model_info['model'] = model.cpu()
        model_info['y_pred_prob'] = total_pred_prob
        print(''.rjust(2),'Complete! Testing time:', '{:.3f}'.format(elapsed_time),'sec')
        print('-'*50) 
        model_result[ml_model_name] =model_info

    return model_result 


