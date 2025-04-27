import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
import utils.metrics as metrics
from hausdorff import hausdorff_distance
import utils.globalvar
from medpy import metric
import time
import torch.nn.functional as F
import random
from utils.lr import *
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, SoftDiceLoss
from utils.loss_functions.atten_matrix_loss import selfsupervise_loss, selfsupervise_loss3
from utils.visualization import visualize_attention_maps, visualize_tokenscore, save_images_and_labels, visualize_token_fusion_with_overlay, visualize_token_fusion_with_overlay_dict,visualize_token_fusion_with_overlay_dict_attnScores,visualize_token_fusion_with_overlay_dict_attnScores_patch16
from einops import rearrange
from thop import profile
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models.model_utils import merge_labelas


#  ============================== add the seed to make sure the results are reproducible ==============================

seed_value = 5000  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution

#  ================================================ parameters setting ================================================

parser = argparse.ArgumentParser(description='DTMFormer')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=600, type=int, metavar='N', help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='batch size (default: 4)')
parser.add_argument('--learning_rate', default=5e-4, type=float, metavar='LR', help='更改后 learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--dataset', default='../../dataset/ACDC/', type=str)
parser.add_argument('--modelname', default='test', type=str, help='type of model')
parser.add_argument('--classes', type=int, default=4, help='number of classes')
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str, help='turn on img augmentation (default: False)')
parser.add_argument('--load', default=False, help='load a pretrained model')
parser.add_argument('--loaddirec', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save', default='default', type=str, help='save the model')
parser.add_argument('--savepath', default='checkpoint1', type=str, help='type of model')
parser.add_argument('--direc', default='./medt', type=str, help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=256)
parser.add_argument('--patchsize', type=int, default=8)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='yes', type=str)
parser.add_argument('--tensorboard', default='./tensorboard_ACDC/', type=str)
parser.add_argument('--eval_mode', default='patient', type=str)
parser.add_argument('--visualize_attn', type=bool, default=True, help='Set this flag to visualize attention maps.')
parser.add_argument('--visualize_tokenscore', type=bool, default=False, help='Set this flag to visualize tokenscore maps.')
parser.add_argument('--visualize_attn_path', type=str, default='/home/wzh/code/UTSFormer/visual/ACDC/attnmap/', help="Path to save attention maps")
parser.add_argument('--visualize_tokenscore_path', type=str, default='/home/wzh/code/UTSFormer/visual/ACDC/tokenscore/', help="Path to save tokenscore maps")
parser.add_argument('--visualize_tokenfusion_path', type=str, default='/home/wzh/code/UTSFormer/visual/ACDC/tokenfusion/', help="Path to save attention maps")



#  =============================================== model initialization ===============================================

args = parser.parse_args()
direc = args.direc  # the path of saving model
eval_mode = args.eval_mode

if args.gray == "yes":
    from utils.utils_multi import JointTransform2D, ImageToImage2D
    imgchant = 1
else:
    from utils.utils_multi_rgb import JointTransform2D, ImageToImage2D
    imgchant = 3
if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None


tf_train = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0, p_contr=0.0, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
tf_val = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_gama=0, color_jitter_params=None, long_mask=True)  # image reprocessing
train_dataset = ImageToImage2D(args.dataset, 'trainofficial', tf_train, args.classes)  # only random horizontal flip, return image, mask, and filename
val_dataset = ImageToImage2D(args.dataset, 'valofficial', tf_val, args.classes)  # no flip, return image, mask, and filename
test_dataset = ImageToImage2D(args.dataset, 'testofficial', tf_val, args.classes)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device("cuda")



from models import SETR, FAT, Transfuse, Transunet_sparse, Transunet

if args.modelname.startswith('Setr'):
    model = SETR.Setr_UTSFormer(in_chans=1, classes=args.classes)
elif args.modelname.startswith('FAT'):
    model = FAT.FAT_Net_DTMFormerV2FirstStageChoose(n_channels=1, n_classes=args.classes)
elif args.modelname.startswith('TransFuse'):
    model = Transfuse.TransFuse_DTMFormerV2FirstStageChoose2(in_channels=1, img_size=args.imgsize, num_classes=args.classes)
elif args.modelname.startswith('TransUnet'):
    model = Transunet_sparse.TransUnet_DTMFormerV2FirstStageChoose(num_classes=args.classes, img_size=args.imgsize)
elif args.modelname.startswith('Compare_Setr0'):
    model = SETR.Setr(n_classes=args.classes)
elif args.modelname.startswith('Compare_FAT0'):
    model = FAT.FAT_Net(n_channels=1, n_classes=args.classes)
elif args.modelname.startswith('Compare_TransFuse0'):
    model = Transfuse.TransFuse(in_channels=1, img_size=args.imgsize, num_classes=args.classes)
elif args.modelname.startswith('Compare_TransUnet0'):
    model = Transunet.TransUnet_Model(num_classes=args.classes, img_size=args.imgsize)
else:
    raise ValueError(f"Model name {args.modelname} is not recognized")







# model = nn.DataParallel(model)
model.to(device)

if args.loaddirec != 'default':
    print(f"Load model: {args.loaddirec}")
    model.load_state_dict(torch.load(args.loaddirec),False)

loss_fn = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
loss_SmoothL1 = nn.SmoothL1Loss()
loss_attns = selfsupervise_loss()
# loss_attns2 = selfsupervise_loss2()
loss_attns3 = selfsupervise_loss3()


# input = torch.randn(1, 1, 256, 256).cuda()
# flops, params = profile(model, inputs=(input, ))
# print("Total GFLOPS: {}".format(flops/1e9))
# print("Total P: {}M".format(params/1e6))



if args.modelname == 'test':
    timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
    boardpath = args.tensorboard + 'pycharm/' + timestr + '_' + args.modelname
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)
    TensorWriter = SummaryWriter(boardpath)
else:
    timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
    boardpath = args.tensorboard + timestr + '_' + args.modelname
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)
    TensorWriter = SummaryWriter(boardpath)

#  ============================================= begin to train the model =============================================
best_dice = 0.0
epoch_losses_train = []
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=1e-5)

torch.autograd.set_detect_anomaly(True)


for epoch in range(args.epochs):
    #  ---------------------------------- training ----------------------------------
    model.train()
    batch_losses = []
    batch_losses1 = []
    batch_losses2 = []
    batch_losses3 = []
    batch_losses4 = []


    current_time = time.time()
    for step, (imgs, label_imgs, img_ids) in enumerate(dataloader):

        imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda()  # (shape: (batch_size, img_h, img_w))


        if args.modelname.startswith('Compare'):
            outputs = model(imgs)
            # compute the loss:
            loss1 = loss_fn(outputs, label_imgs)
            loss2 = torch.tensor(0)
            loss3 = torch.tensor(0)
            loss4 = torch.tensor(0)
            loss = loss1

        else:
            outputs, attnScores, attns, outs, variances, score = model(imgs) # 对应模型 Setr_DTMFormerV2FirstStageChoose

            label_as = rearrange(label_imgs, 'b (h p1) (w p2) -> b h w (p1 p2)', h=int(args.imgsize / args.patchsize),
                                 w=int(args.imgsize / args.patchsize), p1=args.patchsize, p2=args.patchsize).max(
                dim=-1).values
            label_as = (label_as != 0).float()

            # compute the loss:
            loss1 = loss_fn(outputs, label_imgs)
            if args.patchsize == 8:
                loss2 = loss_SmoothL1(attnScores[0], label_as)*0.8 + loss_SmoothL1(attnScores[1], label_as)*0.1 + loss_SmoothL1(attnScores[2], label_as)*0.1
            elif args.patchsize == 16:
                loss2 = loss_SmoothL1(attnScores[0], label_as)*0.8 + loss_SmoothL1(attnScores[1], label_as)*0.1
            else:
                print("loss fail")
            loss3 = loss_attns3(attns)

            loss = loss1 + 0.2 * loss2 + 0.01 * loss3


        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        loss_value1 = loss1.data.cpu().numpy()
        loss_value2 = loss2.data.cpu().numpy()
        loss_value3 = loss3.data.cpu().numpy()

        batch_losses1.append(loss_value1)
        batch_losses2.append(loss_value2)
        batch_losses3.append(loss_value3)


        # -------------------------------------------------------------------


        # optimization step:
        optimizer.zero_grad()  # (reset gradients)
        loss.backward()  # (compute gradients)
        optimizer.step()  # (perform optimization step)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)

    epoch_loss1 = np.mean(batch_losses1)
    epoch_loss2 = np.mean(batch_losses2)
    epoch_loss3 = np.mean(batch_losses3)

    print("train loss: %g" % epoch_loss)
    print("train time: %g" % (time.time() - current_time))
    TensorWriter.add_scalar('train_loss', epoch_loss, epoch)
    TensorWriter.add_scalar('loss1', epoch_loss1, epoch)
    TensorWriter.add_scalar('loss2', epoch_loss2, epoch)
    TensorWriter.add_scalar('loss3', epoch_loss3, epoch)



    #  ----------------------------------- evaluate -----------------------------------
    val_loss = 0
    dices = 0
    hds = 0
    smooth = 1e-25
    mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
    flag = np.zeros(2000)  # record the patients

    model.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []

    current_time = time.time()
    for batch_idx, (imgs, mask, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        imgs = Variable(imgs.to(device='cuda'))
        mask = Variable(mask.to(device='cuda'))

        if args.modelname.startswith('Compare'):
            with torch.no_grad():
                outputs = model(imgs)
        else:
            with torch.no_grad():
                outputs, attnScores, attns, outs, variances, score = model(imgs)

        outputs = F.softmax(outputs, dim=1)
        loss = loss_fn(outputs, mask)
        val_loss = loss.data.cpu().numpy()

        gt = mask.detach().cpu().numpy()
        pred = outputs.detach().cpu().numpy()  # (b, c,h, w) tep
        seg = np.argmax(pred, axis=1)  # (b, h, w) whether exist same score?

        patientid = int(image_filename[:3])
        if flag[patientid] == 0:
            if np.sum(flag) > 0:  # compute the former result
                b, s, h, w = seg_patient.shape
                for i in range(1, args.classes):
                    pred_i = np.zeros((b, s, h, w))
                    pred_i[seg_patient == i] = 1
                    gt_i = np.zeros((b, s, h, w))
                    gt_i[gt_patient == i] = 1
                    mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
                    del pred_i, gt_i
            seg_patient = seg[:, None, :, :]
            gt_patient = gt[:, None, :, :]
            flag[patientid] = 1
        else:
            seg_patient = np.concatenate((seg_patient, seg[:, None, :, :]), axis=1)
            gt_patient = np.concatenate((gt_patient, gt[:, None, :, :]), axis=1)
        # ---------------the last patient--------------
    b, s, h, w = seg_patient.shape
    for i in range(1, args.classes):
        pred_i = np.zeros((b, s, h, w))
        pred_i[seg_patient == i] = 1
        gt_i = np.zeros((b, s, h, w))
        gt_i[gt_patient == i] = 1
        mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
        del pred_i, gt_i
    patients = np.sum(flag)
    mdices = mdices / patients
    for i in range(1, args.classes):
        dices += mdices[i]
    print('epoch [{}/{}], test loss:{:.4f}'.format(epoch, args.epochs, val_loss / (batch_idx + 1)))
    print('epoch [{}/{}], test dice:{:.4f}'.format(epoch, args.epochs, dices / (args.classes - 1)))
    print("val time: %g" % (time.time() - current_time))
    TensorWriter.add_scalar('val_loss', val_loss / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('dices', dices / (args.classes - 1), epoch)

    save_dir = './checkpoints_ACDC/%s/' % args.savepath
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if dices / (args.classes - 1) > best_dice or epoch == args.epochs - 1:
        best_dice = dices / (args.classes - 1)
        timestr = time.strftime('%m%d%H%M')
        save_path = save_dir + '%s_' % timestr + args.modelname + '_%s' % epoch + '_' + str(best_dice)
        torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch != args.epochs - 1:
            checkpoint_best_path = save_dir + args.modelname + '_'  + 'checkpoint_best.pth'
            torch.save(model.state_dict(), checkpoint_best_path, _use_new_zipfile_serialization=False)

    print("lr: ", optimizer.param_groups[0]['lr'])
    TensorWriter.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    optimizer.param_groups[0]['lr'] = poly_lr(epoch, args.epochs - 1, args.learning_rate, 0.9)


checkpoint_best_path = './checkpoints_ACDC/%s/' % args.savepath + args.modelname + '_'  + 'checkpoint_best.pth'
model.load_state_dict(torch.load(checkpoint_best_path))
model.eval()

dices = 0
hds = 0
ious = 0
ses = 0
sps = 0
accs, fs, ps, rs = 0, 0, 0, 0
times = 0
smooth = 1e-25
mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
mses, msps, mious = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes)
maccs, mfs, mps, mrs = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes)

output_file_path = './checkpoints_ACDC/%s/' % args.savepath + args.modelname + 'checkpoint_best.txt'
output_dir = os.path.dirname(output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file_path, 'a') as outfile:
    if eval_mode == "slice":
        for batch_idx, (imgs, mask, *rest) in enumerate(testloader):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = f"{str(batch_idx + 1).zfill(3)}.png"

            test_img_path = os.path.join(args.dataset, 'img', image_filename)
            from utils.imgname import keep_img_name
            keep_img_name(test_img_path)

            imgs = Variable(imgs.to(device='cuda'))
            mask = Variable(mask.to(device='cuda'))

            with torch.no_grad():
                outputs = model(imgs)
            y_out = F.softmax(outputs, dim=1)

            gt = mask.detach().cpu().numpy()
            pred = y_out.detach().cpu().numpy()
            seg = np.argmax(pred, axis=1)
            b, h, w = seg.shape
            for i in range(1, args.classes):
                pred_i = np.zeros((b, h, w))
                pred_i[seg == i] = 255
                gt_i = np.zeros((b, h, w))
                gt_i[gt == i] = 255
                mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
                mhds[i] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
                se, sp, iou, acc, f, precision, recall = metrics.sespiou_coefficient2(pred_i, gt_i)
                maccs[i] += acc
                mfs[i] += f
                mps[i] += precision
                mrs[i] += recall

                mses[i] += se
                msps[i] += sp
                mious[i] += iou
                del pred_i, gt_i

            if args.visual:
                img_ori = cv2.imread(os.path.join(args.dataset, 'img', image_filename))
                img = np.zeros((h, w, 3), dtype=np.uint8)
                img_r, img_g, img_b = img_ori[:, :, 0].copy(), img_ori[:, :, 1].copy(), img_ori[:, :, 2].copy()
                table = np.array([
                    [193, 182, 255], [219, 112, 147], [237, 149, 100],
                    [211, 85, 186], [204, 209, 72], [144, 255, 144],
                    [0, 215, 255], [96, 164, 244], [128, 128, 240],
                    [250, 206, 135]
                ])
                seg0 = seg[0, :, :]

                for i in range(1, args.classes):
                    img_r[seg0 == i] = table[i - 1, 0]
                    img_g[seg0 == i] = table[i - 1, 1]
                    img_b[seg0 == i] = table[i - 1, 2]

                img[:, :, 0] = img_r
                img[:, :, 1] = img_g
                img[:, :, 2] = img_b

                dice_pic = np.zeros(args.classes)
                for i in range(1, args.classes):
                    pred_i = np.zeros((b, h, w))
                    pred_i[seg == i] = 255
                    gt_i = np.zeros((b, h, w))
                    gt_i[gt == i] = 255
                    dice_pic[i] = metrics.dice_coefficient(pred_i, gt_i)
                dice = dice_pic.sum() / (args.classes - 1)
                num = f"{utils.globalvar.CNT:06d}"

                fulldir = os.path.join(args.direc, "")
                os.makedirs(fulldir, exist_ok=True)
                image_filename_new = f"{num}_{image_filename.replace('.png', '')}_{dice}.png"
                cv2.imwrite(os.path.join(fulldir, image_filename_new), img)
            utils.globalvar.CNT += 1

        mdices /= (batch_idx + 1)
        mhds /= (batch_idx + 1)
        mses /= (batch_idx + 1)
        msps /= (batch_idx + 1)
        mious /= (batch_idx + 1)

        maccs /= (batch_idx + 1)
        mfs /= (batch_idx + 1)
        mps /= (batch_idx + 1)
        mrs /= (batch_idx + 1)

        for i in range(1, args.classes):
            dices += mdices[i]
            hds += mhds[i]
            ses += mses[i]
            sps += msps[i]
            ious += mious[i]

            accs += maccs[i]
            fs += mfs[i]
            ps += mps[i]
            rs += mrs[i]

        # 将结果写入文件
        outfile.write(f"mdices: {mdices.tolist()}\n")
        outfile.write(f"mhds: {mhds.tolist()}\n")
        outfile.write(f"mses: {mses.tolist()}\n")
        outfile.write(f"msps: {msps.tolist()}\n")
        outfile.write(f"mious: {mious.tolist()}\n")

        outfile.write(f"Average Dices: {dices / (args.classes - 1)}\n")
        outfile.write(f"Average HDS: {hds / (args.classes - 1)}\n")
        outfile.write(f"Average SES: {ses / (args.classes - 1)}\n")
        outfile.write(f"Average SPS: {sps / (args.classes - 1)}\n")
        outfile.write(f"Average IOUs: {ious / (args.classes - 1)}\n")

        outfile.write(f"Accuracies: {accs / (args.classes - 1)}\n")
        outfile.write(f"F-scores: {fs / (args.classes - 1)}\n")
        outfile.write(f"Precisions: {ps / (args.classes - 1)}\n")
        outfile.write(f"Recalls: {rs / (args.classes - 1)}\n")
        outfile.write(f"Times: {times}\n")

        print(mdices, '\n', mhds, '\n', mses, '\n', msps, '\n', mious, '\n')
        print(dices / (args.classes - 1), hds / (args.classes - 1), ses / (args.classes - 1), sps / (args.classes - 1),
              ious / (args.classes - 1))
        print(accs / (args.classes - 1), fs / (args.classes - 1), ps / (args.classes - 1), rs / (args.classes - 1))
        print(times)

    else:
        flag = np.zeros(2000)
        times = 0
        mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
        mses, msps, mious = np.zeros(args.classes), np.zeros(args.classes), np.zeros(args.classes)
        current_time = time.time()

        for batch_idx, (imgs, mask, *rest) in enumerate(testloader):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = f"{str(batch_idx + 1).zfill(3)}.png"

            imgs = Variable(imgs.to(device='cuda'))
            mask = Variable(mask.to(device='cuda'))

            if args.modelname.startswith('Compare'):
                with torch.no_grad():
                    outputs = model(imgs)
            else:
                with torch.no_grad():
                    outputs, attnScores, attns, outs, variances, score = model(
                        imgs)

                img_name = os.path.splitext(image_filename)[0]
                save_dir = os.path.join(args.visualize_tokenfusion_path, args.modelname, f"test")


            y_out = F.softmax(outputs, dim=1)

            gt = mask.detach().cpu().numpy()
            pred = y_out.detach().cpu().numpy()
            seg = np.argmax(pred, axis=1)

            patientid = int(image_filename[:3])
            if flag[patientid] == 0:
                if np.sum(flag) > 0:  # compute the former result
                    b, s, h, w = seg_patient.shape
                    for i in range(1, args.classes):
                        pred_i = np.zeros((b, s, h, w))
                        pred_i[seg_patient == i] = 1
                        gt_i = np.zeros((b, s, h, w))
                        gt_i[gt_patient == i] = 1
                        mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
                        if pred_i.sum() != 0 and gt_i.sum() != 0:
                            mhds[i] += metric.binary.hd95(pred_i, gt_i)
                        se, sp, iou, acc, f, precision, recall = metrics.sespiou_coefficient2(pred_i, gt_i)
                        maccs[i] += acc
                        mfs[i] += f
                        mps[i] += precision
                        mrs[i] += recall
                        mses[i] += se
                        msps[i] += sp
                        mious[i] += iou
                        del pred_i, gt_i
                seg_patient = seg[:, None, :, :]
                gt_patient = gt[:, None, :, :]
                flag[patientid] = 1
            else:
                seg_patient = np.concatenate((seg_patient, seg[:, None, :, :]), axis=1)
                gt_patient = np.concatenate((gt_patient, gt[:, None, :, :]), axis=1)

            utils.globalvar.CNT += 1

        b, s, h, w = seg_patient.shape
        for i in range(1, args.classes):
            pred_i = np.zeros((b, s, h, w))
            pred_i[seg_patient == i] = 1
            gt_i = np.zeros((b, s, h, w))
            gt_i[gt_patient == i] = 1
            mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
            if pred_i.sum() != 0 and gt_i.sum() != 0:
                mhds[i] += metric.binary.hd95(pred_i, gt_i)
            se, sp, iou, acc, f, precision, recall = metrics.sespiou_coefficient2(pred_i, gt_i)
            maccs[i] += acc
            mfs[i] += f
            mps[i] += precision
            mrs[i] += recall

            mses[i] += se
            msps[i] += sp
            mious[i] += iou
            del pred_i, gt_i

        patients = np.sum(flag)
        mdices /= patients
        mhds /= patients
        mses /= patients
        msps /= patients
        mious /= patients

        maccs /= patients
        mfs /= patients
        mps /= patients
        mrs /= patients

        print("--------------------------------")
        print("mdices, mhds, mses, msps, mious")
        print(mdices)
        print(mhds)
        print(mses)
        print(msps)
        print(mious)

        print("--------------------------------")
        print("maccs, mfs, mps, mrs")
        print(maccs)
        print(mfs)
        print(mps)
        print(mrs)
        print("--------------------------------")

        for i in range(1, args.classes):
            dices += mdices[i]
            hds += mhds[i]
            ious += mious[i]
            ses += mses[i]
            sps += msps[i]

            accs += maccs[i]
            fs += mfs[i]
            ps += mps[i]
            rs += mrs[i]

        print("dices, hds, ious, ses, sps")
        print(dices / (args.classes - 1))
        print(hds / (args.classes - 1))
        print(ious / (args.classes - 1))
        print(ses / (args.classes - 1))
        print(sps / (args.classes - 1))
        print("accs, fs, ps, rs")
        print(accs / (args.classes - 1))
        print(fs / (args.classes - 1))
        print(ps / (args.classes - 1))
        print(rs / (args.classes - 1))

        print("val time: %g" % (time.time() - current_time))

        # 将结果写入文件
        outfile.write("--------------------------------\n")
        outfile.write("mdices, mhds, mses, msps, mious\n")
        outfile.write(f"{mdices.tolist()}\n")
        outfile.write(f"{mhds.tolist()}\n")
        outfile.write(f"{mses.tolist()}\n")
        outfile.write(f"{msps.tolist()}\n")
        outfile.write(f"{mious.tolist()}\n")

        outfile.write("--------------------------------\n")
        outfile.write("maccs, mfs, mps, mrs\n")
        outfile.write(f"{maccs.tolist()}\n")
        outfile.write(f"{mfs.tolist()}\n")
        outfile.write(f"{mps.tolist()}\n")
        outfile.write(f"{mrs.tolist()}\n")
        outfile.write("--------------------------------\n")

        outfile.write("dices, hds, ious, ses, sps\n")
        outfile.write(f"{dices / (args.classes - 1)}\n")
        outfile.write(f"{hds / (args.classes - 1)}\n")
        outfile.write(f"{ious / (args.classes - 1)}\n")
        outfile.write(f"{ses / (args.classes - 1)}\n")
        outfile.write(f"{sps / (args.classes - 1)}\n")

        outfile.write("accs, fs, ps, rs\n")
        outfile.write(f"{accs / (args.classes - 1)}\n")
        outfile.write(f"{fs / (args.classes - 1)}\n")
        outfile.write(f"{ps / (args.classes - 1)}\n")
        outfile.write(f"{rs / (args.classes - 1)}\n")

        outfile.write(f"val time: {time.time() - current_time}\n")