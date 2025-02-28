from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34,resnet50,resnet18
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import Each_labelloss
from sklearn.metrics import roc_auc_score

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(0)
# print(torch.cuda.get_device_name(0))
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=20000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--Analysis', type=str, default='',
                    help='select analysis parameters for Gate')
parser.add_argument('--method', type=str, default='All_Loss',
                    choices=['All_Loss'],
                    )
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.5, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--thred', type=float, default=0.8, metavar='T',
                    help='thred for pseduo (default: 0.5)')
parser.add_argument('--lambda_WAL', type=float, default=0.1, metavar='LAM',
                    help='lamda for WAL')
parser.add_argument('--lambda_CLA', type=float, default=0.05, metavar='LAM',
                    help='lamda for CLA')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model/save_model_DA_FeaGate',
                    help='dir to save checkpoint')
parser.add_argument('--checkpath_step', type=str, default='./save_model/save_model',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--inc', type=int, default=512,
                    help='which network to use')
parser.add_argument('--source', type=str, default='PA_MultiLabel',
                    help='source domain')
parser.add_argument('--target', type=str, default='AP_MultiLabel',
                    help='target domain')

parser.add_argument('--dataset', type=str, default='Chex14ray',
                    choices=['Chex14ray','Chexpert'],
                    help='the name of dataset')
parser.add_argument('--bs', type=int, default=64,
                    help='batch_size')
parser.add_argument('--img_size', type=int, default=256,
                    help='img_size')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=10, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=False,
                    help='early stopping on validation or not')
parser.add_argument('--Class_N', type=int, default=14,
                    help='number of labeled examples in the target')


args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
source_loader,target_loader, target_loader_test, class_list = return_dataset(args)
class_list=[i for i in range(args.Class_N)]
use_gpu = torch.cuda.is_available()
print(use_gpu)
record_dir = 'record_DA2_FeaGate/%s/%s_multi/%s_%d' % (args.Analysis,args.dataset, args.method,args.bs)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)


torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = args.inc
elif args.net == 'resnet50':
    G = resnet50()
    inc = args.inc
elif args.net == 'resnet18':
    G = resnet18()
    inc = args.inc
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr':args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                    inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)

weights_init(F1)
lr = args.lr
G.to(DEVICE)
F1.to(DEVICE)


im_data_s = torch.FloatTensor(1)
im_data_s2 = torch.FloatTensor(1)
gt_labels_s = torch.FloatTensor(1)
gt_labels_s2 = torch.FloatTensor(1)
im_data_t=torch.FloatTensor(1)
gt_labels_t=torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)


im_data_s = im_data_s.to(DEVICE)
im_data_s2 = im_data_s2.to(DEVICE)
gt_labels_s = gt_labels_s.to(DEVICE)
gt_labels_s2= gt_labels_s2.to(DEVICE)
im_data_t=im_data_t.to(DEVICE)
gt_labels_t=gt_labels_t.to(DEVICE)
im_data_tu = im_data_tu.to(DEVICE)


im_data_s = Variable(im_data_s)
im_data_s2= Variable(im_data_s2)
gt_labels_s = Variable(gt_labels_s)
gt_labels_s2 = Variable(gt_labels_s2)
im_data_t=Variable(im_data_t)
gt_labels_t=Variable(gt_labels_t)
im_data_tu = Variable(im_data_tu)


if os.path.exists(args.checkpath) == False:
    os.makedirs(args.checkpath)


def train():
    record_file = os.path.join(record_dir,
                               '%s_test1_%s_%s_to_%s_lam_%s_lamea_%s.txt' %
                               (args.method, args.net, args.source,
                                args.target,args.lambda_WAL,args.lambda_CLA))

    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)


    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    counter = 0
    best_acc_test=0
    with open(record_file, 'a') as f:
        f.write(
            '################  {} Feature Gate  Source {} to {} bs {} for learning rate of {},lr: g :{:.5f}, f:{:.5f},with lambda {}, label_Each lambda {}  thred {} T {} SGD ######################## \n'.format(
                args.method, args.source, args.target, args.bs, args.lr,param_lr_g[0], param_lr_f[0],  args.lambda_WAL,args.lambda_CLA,args.thred,args.T))  # Discriminator,dc is leakyRelu




    for step in range (all_step):

        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)


        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)

        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        im_data_s.data=im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.data=gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.data=im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.data=gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.data=im_data_tu.data.resize_(data_t[0].size()).copy_(data_t[0])

        zero_grad_all()
        data=im_data_s
        label=gt_labels_s
        output_G = G(data)
        out_all = F1(output_G)
        loss_cls = criterion(out_all, label)
        loss=loss_cls
        output = G(im_data_tu)
        loss_t = Each_labelloss(F1, output_G, output, label, args.lambda_WAL, args.lambda_CLA, args.thred, args.T,
                                eta=1.0) + loss
        loss_t.backward()
        optimizer_f.step()
        optimizer_g.step()


        log_train = 'S {} T {} Train Ep: {} lr{:.6f} \t ' \
                    'Loss Classification: {:.6f} Loss T {:.6f} ' \
                    'Method {}\n'.format(args.source, args.target,
                                         step, lr, loss.data,
                                         -loss_t.data, args.method)

        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            print(step)
            loss_test, acc_test,average_auc_mi,average_auc_ma = test(target_loader_test)
            #
            loss_train,acc_train,average_auc_mi_train,average_auc_ma_train=test(source_loader)

            if acc_test >= best_acc_test:
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1

            print('best acc test %f now test %f  train source acc %f ' % (best_acc_test,acc_test,
                                                        acc_train))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d cls loss %.3f  best acc %.3f,  '
                        'now target loss %.4f acc %.3f,oAUC %.3f,  cAUC %.3f '
                        'train source loss %.4f acc %.3f, oAUC %.3f ,cAUC %.3f,\n' % (step,loss_cls,  best_acc_test,
                                                         loss_test,acc_test,average_auc_mi,average_auc_ma,
                                                         loss_train,acc_train,average_auc_mi_train,average_auc_ma_train))

            G.train()
            F1.train()
            if args.early:
                if counter > args.patience:
                    break

            if args.save_check:
                save_path=args.checkpath_step
                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                torch.save(G.state_dict(),
                           os.path.join(save_path,
                                        "G_iter_model_{}_{}_to_{}"
                                        "_step_{}.pth.tar".
                                        format(args.method, args.source,args.target,
                                               step)))
                torch.save(F1.state_dict(),
                           os.path.join(save_path,
                                        "F1_iter_model_{}_{}_to_{}"
                                        "_step_{}.pth.tar".
                                        format(args.method, args.source,args.target
                                               ,step)))

    loss_test, acc_test, average_auc_mi, average_auc_ma, = test(target_loader_test)
    #
    loss_train, acc_train, average_auc_mi_train,average_auc_ma_train, = test(source_loader)
    print('Final  now test %f  train source acc %f  ' % (acc_test,
                                                         acc_train))
    print('record %s' % record_file)
    with open(record_file, 'a') as f:
        f.write('Final  best acc %.3f,  '
                'now target loss %.4f acc %.3f,oAUC %.3f, cAUC %.3f '
                'train source loss %.4f acc %.3f, oAUC %.3f , cAUC %.3f,\n' % (  best_acc_test,
                                                                         loss_test, acc_test, average_auc_mi,
                                                                         average_auc_ma,loss_train, acc_train, average_auc_mi_train,
                                                                         average_auc_ma_train))
    if args.save_check:
        path_special="Predictor_C"
        args.checkpath=args.checkpath
        print('saving model')
        torch.save(G.state_dict(),
                   os.path.join(args.checkpath,
                                "G_iter_model_{}_{}"
                                "spe_{}.pth.tar".
                                format(args.method, args.source,
                                       path_special)))
        torch.save(F1.state_dict(),
                   os.path.join(args.checkpath,
                                "F1_iter_model_{}_{}"
                                "spe_{}.pth.tar".
                                format(args.method, args.source,
                                       path_special)))




def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    Correct_matrix=np.zeros(num_class)

    output_all = np.zeros((0, num_class))
    label_all=np.zeros((0, num_class))
    pred_all=np.zeros((0, num_class))
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    confusion_matrix = torch.zeros(num_class, num_class)
    average_auc=0
    aucs = []
    with torch.no_grad():
        average_auc_mi,average_auc_ma,count=0,0,0
        for batch_idx, data_t in enumerate(loader):
            im_data_t.data=im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data=gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])

            feat = G(im_data_t)
            output1 = F1(feat)
            output2=nn.functional.sigmoid(output1)
            output_all = np.r_[output_all, output2.data.cpu().numpy()]
            label_all = np.r_[label_all, gt_labels_t.data.cpu().numpy()]

            size+= im_data_t.size(0)*gt_labels_t.data.shape[1]
            pred1=torch.where(output2.data >= 0.5, 1, 0)
            pred_all=np.r_[pred_all, pred1.data.cpu().numpy()]
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)




    average_auc_mi = roc_auc_score(label_all, output_all, average="micro")
    average_auc_ma = roc_auc_score(label_all, output_all, average="macro")


    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%) \n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size,average_auc_mi,average_auc_ma






train()

