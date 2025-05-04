import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import MyDataSet
from utils import read_train_data, read_val_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, test
from glob import glob
import pandas as pd
from lib import *
from tqdm import tqdm
from metrics import precision_and_recall, dice_score, sensitivity_and_specificity, print_comprehensive_score, print_and_get_comprehensive_score
from sklearn.utils import shuffle

def get_all_dir(directory_path):
    res = []
    # 循环遍历这些条目，检查哪些是文件夹
    for dir_name in os.listdir(directory_path):
        if dir_name.startswith(".ipynb_checkpoints"):
            continue
        # 使用os.path.join来获取完整的路径
        full_path = os.path.join(directory_path, dir_name)
        # 检查这个路径是否是一个目录
        if os.path.isdir(full_path):
            res.append(dir_name)
    return res


def get_csv_data(file_path):
    # 加载 Excel 文件
    df = pd.read_csv(file_path)
    # 显示 DataFrame 的前几行，以确认正确读取

    # 提取特定的列
    id_evaluation_list = df[['Patient ID', 'Radiological Evaluation']]

    # category_id - 1  Mild=0, Moderate=1, Severe=2
    id_evaluation_dict = {}
    for index, row in id_evaluation_list.iterrows():
        id_evaluation_dict[str(row['Patient ID'])] = row['Radiological Evaluation'] - 1

    # print(id_evaluation_dict)

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
    t2_csv_file_path = os.path.join(args.data_root, 'Metadata', 'T2_age_gender_evaluation.csv')
    t2_images_ids, t2_id_evaluation_dict = get_csv_data(t2_csv_file_path)
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

    print(f"train_patients: {len(train_names)}, valid_patients: {len(valid_names)}, test_patients: {len(test_names)}")
    print(f"train_patients: {train_names}")
    print(f"valid_patients: {valid_names}")
    print(f"test_patients: {test_names}")

    """ Training data """
    train_x, train_y = [], []
    for name in train_names:
        if name in paired_id_evaluation_dict.keys():
            x, y = get_data(train_path, name, paired_id_evaluation_dict)
            train_x += x
            train_y += y

    """ Validation data """
    valid_x, valid_y = [], []
    for name in valid_names:
        if name in paired_id_evaluation_dict.keys():
            x, y = get_data(valid_path, name, paired_id_evaluation_dict)
            valid_x += x
            valid_y += y

    """ Testing data """
    test_x, test_y = [], []
    for name in test_names:
        if name in paired_id_evaluation_dict.keys():
            x, y = get_data(test_path, name, paired_id_evaluation_dict)
            test_x += x
            test_y += y

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


def main(args):
    build_model = f"build_{args.model_name}"
    save_path = f"{args.save_dir}/{args.model_name}"
    checkpoint_path = f'./{save_path}/model_weight/best_model.pth'

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    print(f"test model: {args.model_name}")
    print(args)

    (train_images_path, train_images_label), (val_images_path, val_images_label), (test_images_path, test_images_label) = load_CirrMRI_classification_data(args)
    data_str = f"Dataset Size:\nTrain: {len(train_images_path)} - Valid: {len(val_images_path)} - Test: {len(test_images_path)}"
    print(data_str)

    data_transform = transforms.Compose(
        [
         transforms.Resize((args.img_size, args.img_size)),
        #  transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    test_dataset = MyDataSet(images_path=test_images_path,
                              images_class=test_images_label,
                              transform=data_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=test_dataset.collate_fn)

    model = eval(build_model)(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.eval()

    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    all_preds = []
    all_labels = []
    for step, data in enumerate(test_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_acc =  accu_num.item() / sample_num
    print("[test] size: {} accuracy: {}".format(len(test_loader), round(test_acc, 3)))
    precision_macro, recall_macro, specificity_macro, f1_macro, precision, sensitivity, specificity, f1 = print_and_get_comprehensive_score(
        all_labels, all_preds)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    print(
        f"overall: &{test_acc:.3f} &{total / 1e6:.2f} &{precision_macro * 100:.2f} &{recall_macro * 100:.2f} &{specificity_macro * 100:.2f} &{f1_macro * 100:.2f} \\\\")
    stages = ['mild', 'moderate', 'severe']
    for i, stage in enumerate(stages):
        print(
            f"{stage}: &{precision[i] * 100:.2f} &{sensitivity[i] * 100:.2f} &{specificity[i] * 100:.2f} &{f1[i] * 100:.2f} \\\\")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='model name')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--data_root', type=str, default="/root/ZJ/Dataset/liver/CirrMRI600+")
    parser.add_argument('--modality', type=str, default="t2")
    parser.add_argument('--save_dir', type=str, default="files_t2_2d")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
