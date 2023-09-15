from jittor import nn
import time
import argparse
from deeplab import DeepLab
from dataset import *
import numpy as np
from utils import Evaluator
import os
jt.flags.use_cuda = 1

def convert_time(timestamp):
    m, s = divmod(timestamp, 60)
    h, m = divmod(m, 60)
    return h, m, s

def poly_lr_scheduler(opt, init_lr, iter, epoch, max_iter, max_epoch):
    new_lr = init_lr * (1 - float(epoch * max_iter + iter) / (max_epoch * max_iter)) ** 0.9
    opt.lr = new_lr

    # print ("epoch ={} iteration = {} new_lr = {}".format(epoch, iter, new_lr))


def train(model, train_loader, optimizer, epoch, init_lr, log_path, start_time):
    print("Train for Epoch{}...".format(epoch))
    model.train()
    max_iter = len(train_loader)
    total_epoch_loss = 0

    for idx, (image, target, _) in enumerate(train_loader):
        poly_lr_scheduler(optimizer, init_lr, idx, epoch, max_iter, 50)
        image = image.float32()
        pred = model(image)
        target = jt.round(target[:, 1, :, :]).int8()
        loss = nn.cross_entropy_loss(pred, target, ignore_index=255) # fix a bug
        print(loss)
        # writer.add_scalar('train/total_loss_iter', loss.data, idx + max_iter * epoch)
        total_epoch_loss += loss.item()
        optimizer.step (loss)
        # print ('Training in epoch {} iteration {} loss = {}'.format(epoch, idx, loss.data[0]))
        if (idx%5 == 0):
            print("Batch {}/{}".format(idx, len(train_loader)))
    print('Epoch {} loss = {}'.format(epoch, total_epoch_loss/len(train_loader)))
    with open(log_path, 'a+') as f:
        h, m, s = convert_time(time.time() - start_time)
        f.write("\n"+ "%02dh%02dm%02ds" % (h, m, s))
        f.write('\n'+ 'Epoch {} loss = {}'.format(epoch, total_epoch_loss/len(train_loader)))

def val (model, val_loader, epoch, evaluator, log_path, start_time):
    print("Validate for Epoch...".format(epoch))
    model.eval()
    evaluator.reset()
    for idx, (image, target, _) in enumerate(val_loader):
        image = image.float32()
        output = model(image)
        pred = output.data
        target = jt.round(target[:, 1, :, :]).int8()
        target = target.data
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
        # print ('Test in epoch {} iteration {}'.format(epoch, idx))
        if (idx%5 == 0):
            print("Batch {}/{}".format(idx, len(val_loader)))
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    best_miou = 0.0
    # writer.add_scalar('val/mIoU', mIoU, epoch)
    # writer.add_scalar('val/Acc', Acc, epoch)
    # writer.add_scalar('val/Acc_class', Acc_class, epoch)
    # writer.add_scalar('val/fwIoU', FWIoU, epoch)

    if (mIoU > best_miou):
        best_miou = mIoU

    print ('Testing result of epoch {} miou = {} Acc = {} Acc_class = {} \
                FWIoU = {} Best Miou = {}'.format(epoch, mIoU, Acc, Acc_class, FWIoU, best_miou))

    with open(log_path, 'a+') as f:
        h, m, s = convert_time(time.time() - start_time)
        f.write("\n" + "%02dh%02dm%02ds" % (h, m, s))
        f.write('\n'+'Testing result of epoch {} miou = {} Acc = {} Acc_class = {} \
                FWIoU = {} Best Miou = {}'.format(epoch, mIoU, Acc, Acc_class, FWIoU, best_miou))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
    parser.add_argument("--data_path", type=str, default="./datasets")
    parser.add_argument("--output_path", type=str, default="./results/single_gpu")
    parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=384, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--label_c", type=int, default=30, help="number of image channels")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    opt = parser.parse_args()

    model = DeepLab(output_stride=16, num_classes=29)
    # model.load("./Epoch_40.pkl")
    # train_loader = TrainDataset(data_root='/home/guomenghao/voc_aug/mydata/', split='train', batch_size=4, shuffle=True)
    # val_loader = ValDataset(data_root='/home/guomenghao/voc_aug/mydata/', split='val', batch_size=1, shuffle=False)

    print("Loading datasets...")
    
    transform_label = [
        transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.NEAREST),
        transform.ToTensor(),
        # transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    transform_img = [
        transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]

    train_loader = ImageDataset(opt.data_path, transform_label=transform_label, transform_img=transform_img, mode="train").set_attrs(
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_loader = ImageDataset(opt.data_path, transform_label=transform_label, transform_img=transform_img, mode="val").set_attrs(
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )

    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    # writer = SummaryWriter(os.path.join('curve', 'train.events.wo_drop'))
    epochs = 50
    evaluator = Evaluator(29)

    start_time = time.time()
    # log_path = "./log/log{}.txt".format(round(start_time))
    model_path = "./model/model{}/".format(round(start_time))
    log_path = "./model/model{}/Log.txt".format(round(start_time))
    os.mkdir(model_path)
    with open(log_path, 'w') as f:
        f.write("Start at {}".format(start_time))
        print("create log")

    print("Start to train...")

    for epoch in range (epochs):
        # train(model, train_loader, optimizer, epoch, learning_rate, writer)
        # val(model, val_loader, epoch, evaluator, writer)
        train(model, train_loader, optimizer, epoch, learning_rate, log_path, start_time)
        val(model, val_loader, epoch, evaluator, log_path, start_time)
        if (epoch % opt.checkpoint_interval == 0):
            model.save(os.path.join(f"{model_path}Epoch_{epoch + 1}.pkl"))


    total_run_time = time.time() - start_time
    with open(log_path, 'a+') as f:
        h, m, s = convert_time(total_run_time)
        f.write("\n\n" + "Total run time %02dh%02dm%02ds" % (h, m, s))

    print("Training ends")



if __name__ == '__main__' :
    main ()
