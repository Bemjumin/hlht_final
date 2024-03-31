# This is a sample Python script.
import torch
import argparse
from dataset import get_dataSST_2
from performance.nlp_performance import PerformEvaluation as nlp_Per
from performance.performance import PerformEvaluation
from dataset import load_dataloader
from select_imagenetmodel import load_imagenet_model
import warnings
warnings.filterwarnings("ignore")
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

'''
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
'''


def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sst2',
                        help='training dataset (default: sst2)')
    parser.add_argument('--application', type=str, default='nlp',
                        help='application scenario (default: cv)')
    parser.add_argument('--root', default='./model_weights/', type=str, metavar='PATH',
                        help='path to the pretrain model')
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--criterion', type=str, default='cpl')
    parser.add_argument('--batch_size', type=int, default=16)

    # opt = vars(parser.parse_args())     # Object: Namespace -> 字典
    arguments = parser.parse_args()
    return arguments


def test(test_dataloader, device, model):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载模型
    if args.application == 'cv':
        model = load_imagenet_model(args.model, device)
        test_dataloader, total_num = load_dataloader(args.batch_size)
    else:
        train_dataloader, test_dataloader = get_dataSST_2()
        model = load_imagenet_model('bert', device)

    if args.application == 'cv':
        f = PerformEvaluation(model, device, test_dataloader)
        if args.criterion == 'acc':
            print('----------ocp性能测试开始----------')
            f.occupancy_rate()
            print('----------ocp性能测试结束----------')
        elif args.criterion == 'cpl':
            print('----------sparse性能测试开始----------')
            f.compute_sparse_degree()
            print('----------sparse性能测试结束----------')
        elif args.criterion == 'cpl':
            print('----------cpl性能测试开始----------')
            f.computation()
            print('----------cpl性能测试结束----------')
    elif args.application == 'nlp':
        p = nlp_Per(model, device, test_dataloader)
        if args.criterion == 'acc':
            print('----------性能测试开始----------')
            print('----------性能测试结束----------')
        elif args.criterion == 'cpl':
            print('----------性能测试开始----------')
            print('----------性能测试结束----------')
        elif args.criterion == 'ocp':
            print('----------ocp性能测试开始----------')
            p.occupancy_rate()
            print('----------ocp性能测试结束----------')


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    args = collect_args()
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
