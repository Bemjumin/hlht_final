# This is a sample Python script.
import torch
import torch.nn as nn


class FunctionEvaluation:
    def __init__(self, model, device, dataloader):

        """
        @description:
        @param {
            model_bert:需要测试的模型
            device: 设备(GPU)
            dataloader:可迭代的测试数据集
            }
        @return: None
        """
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.init_model(device)

    def init_model(self, device):
        """
        @description:将网络部署到环境设备上
        @param {
            device: 设备（GPU）
        }
        @return: None
        """
        self.model.eval().to(device)
        pass
    '''
    def prop_evaluate(self):
        """
        @description: 检测模型输出类型数量是否符合所用数据集类别数量
        @param: None
        @return: Bool
        """
        num = 0
        for name, layer in reversed(list(self.model.named_modules())):
            if isinstance(layer, nn.Linear):
                num = layer.bias.size(0)
                break
        if len(self.dataloader.dataset.classes) == num:
            print("The model is proper, number of classes is equal to the number of outputs: ", num)
            return True
        else:
            print("The model is not proper, number of classes is not equal to the number of outputs")
            print("The model has {} outputs, but the number of classes is {}".format(num, len(self.dataloader.dataset.classes)))
            return False
    '''

    def complete_evaluate(self):
        """
        @description: 检测架构是否完整
        @return: None
        """
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
        else:
            device = 'cpu'
        print("device:", device)
        print("model:", self.model)
        print("framework: pytorch")

    def acc_evaluate(self):
        """
            @description: 检测模型在给定数据集下的准确率
            @param: None
            @return: float: 准确率（小于1）
        """
        if 1:
            self.model.eval()
            total = 0
            correct = 0
            print('------------start test:-------------')
            with torch.no_grad():
                for data in self.dataloader:
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the dataset is: %d %%' % (
                    100 * correct / total))
            return correct / total
        else:
            raise print('the result does not match, please change check')

    def nlp_acc_eval(self):
        total = 0
        correct = 0
        print('------------start test:-------------')
        with torch.no_grad():
            for data in self.dataloader:
                tid, ttid, mask, labels = data
                tid, ttid, mask, labels = tid.to(self.device), ttid.to(self.device), mask.to(self.device), labels.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                logits = self.model(input_ids=tid, token_type_ids=ttid, attention_mask=mask).logits.argmax(dim=1)
                total += labels.size(0)
                correct += (logits == labels).sum()

        print('Accuracy of the network on the dataset is: %d %%' % (
                100 * correct / total))
        return correct / total