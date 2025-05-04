import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import MyDataSet, print_and_save, create_dir, seeding
from lib import *
from utils import read_train_data, read_val_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from glob import glob
import pandas as pd
import datetime
from sklearn.utils import shuffle

def get_all_dir(directory_path):
    res = []
    for dir_name in os.listdir(directory_path):
        if dir_name.startswith(".ipynb_checkpoints"):
            continue
        full_path = os.path.join(directory_path, dir_name)
        if os.path.isdir(full_path):
            res.append(dir_name)
    return res


def get_csv_data(file_path):
    df = pd.read_csv(file_path)
    id_evaluation_list = df[['Patient ID', 'Radiological Evaluation']]

    id_evaluation_dict = {}
    for index, row in id_evaluation_list.iterrows():
        id_evaluation_dict[str(row['Patient ID'])] = row['Radiological Evaluation'] - 1

    images_ids = df['Patient ID'].tolist()

    return images_ids, id_evaluation_dict


def load_CirrMRI_classification_data(args):
    def get_data(path, name, label_dict):
        images = sorted(glob(os.path.join(path, str(name), "images", "*.png")))
        label = label_dict[name]
        labels = [label] * len(images)
        return images, labels

    t2_data_path = os.path.join(args.data_root, 'Cirrhosis_T2_2D')
    data_path = t2_data_path
    paired_csv_file_path = os.path.join(args.data_root, 'Metadata', 'T1&T2_Paired_age_gender_evaluation.csv')
    paired_images_ids, paired_id_evaluation_dict = get_csv_data(paired_csv_file_path)

    """ Names """
    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")

    train_names = get_all_dir(train_path)
    valid_names = get_all_dir(valid_path)
    test_names = get_all_dir(test_path)

    if args.modality == 't1':
        data_path = os.path.join(args.data_root, 'Cirrhosis_T1_2D_all')
        train_path = data_path
        valid_path = data_path
        test_path = data_path

    train_names = shuffle(train_names, random_state=42)

    train_len, valid_len, test_len = 0, 0, 0
    """ Training data """
    train_x, train_y = [], []
    for name in train_names:
        if name in paired_id_evaluation_dict.keys():
            train_len += 1
            x, y = get_data(train_path, name, paired_id_evaluation_dict)
            train_x += x
            train_y += y

    """ Validation data """
    valid_x, valid_y = [], []
    for name in valid_names:
        if name in paired_id_evaluation_dict.keys():
            valid_len += 1
            x, y = get_data(valid_path, name, paired_id_evaluation_dict)
            valid_x += x
            valid_y += y

    """ Testing data """
    test_x, test_y = [], []
    for name in test_names:
        if name in paired_id_evaluation_dict.keys():
            test_len += 1
            x, y = get_data(test_path, name, paired_id_evaluation_dict)
            test_x += x
            test_y += y

    print(f"train_patients: {train_len}, valid_patients: {valid_len}, test_patients: {test_len}")

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

def main(args):
    """ Seeding """
    seeding(42)

    build_model = f"build_{args.model_name}"
    save_path = f"{args.save_dir}/{args.model_name}"
    weight_path = f"{save_path}/model_weight"

    create_dir(f"{weight_path}/checkpoint")

    """ Training logfile """
    train_log_path = f"{save_path}/train_log.txt"
    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print_and_save(train_log_path, f"using {device} device.")

    print_and_save(train_log_path, str(args))
    print_and_save(train_log_path, 'Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    tb_writer = SummaryWriter(save_path)

    data_str = f"train model: {args.model_name}"
    print_and_save(train_log_path, data_str)

    (train_images_path, train_images_label), (val_images_path, val_images_label), (test_images_path, test_images_label) = load_CirrMRI_classification_data(args)
    data_str = f"Dataset Size:\nTrain: {len(train_images_path)} - Valid: {len(val_images_path)} - Test: {len(test_images_path)}\n"
    print_and_save(train_log_path, data_str)
    data_str = f"Early Stopping Patience: {args.early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                     transforms.RandomHorizontalFlip(0.3),
                                     transforms.RandomVerticalFlip(0.3),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print_and_save(train_log_path, 'Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = eval(build_model)(num_classes=args.num_classes).to(device)

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    best_acc = 0.
    start_epoch = 0
    early_stopping_count = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        lr_scheduler.step(val_loss)


        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), f"{weight_path}/best_model.pth")
            print_and_save(train_log_path, "Saved epoch{} as new best model".format(epoch))
            best_acc = val_acc
            early_stopping_count = 0

        elif best_acc > val_acc:
            early_stopping_count += 1

        #add loss, acc and lr into tensorboard
        print_and_save(train_log_path, "[epoch {}] train accuracy: {}, val accuracy: {},  train loss: {}, val loss: {}".format(epoch, round(train_acc, 3), round(val_acc, 3), round(train_loss, 3), round(val_loss, 3)))

        if early_stopping_count == args.early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {args.early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break


    total = sum([param.nelement() for param in model.parameters()])
    print_and_save(train_log_path, "Number of parameters: %.2fM" % (total/1e6))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='model name')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--data_root', type=str, default="data/CirrMRI600+")
    parser.add_argument('--modality', type=str, default="t2")
    parser.add_argument('--save_dir', type=str, default="files_t2_2d")
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
