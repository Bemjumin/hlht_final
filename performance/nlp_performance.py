import torch
import subprocess
import time


class PerformEvaluation:
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
        self.dataloader = dataloader
        # 使用GPU进行测量
        self.device = device
        self.model.to(self.device)

    # 测量GPU占用率的函数
    def measure_gpu_usage(self):
        """
            @description:进行性能测试一：硬件使用情况
            @param None
            @return: float: ( 1.平均GPU内存占用率 (MB)
                              2.平均GPU利用内存 (%)
                              3.总时间 (秒) )
        """
        self.model.eval()
        total_gpu_memory_usage = 0
        total_gpu_utilization = 0
        start_time = time.time()

        for data in self.dataloader:
            # inputs = inputs.to(self.device)
            tid, ttid, mask, labels = data
            tid, ttid, mask, labels = tid.to(self.device), ttid.to(self.device), mask.to(self.device), labels.to(
                self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)
            # 测量GPU内存占用率
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                # outputs = self.model(inputs)
                logits = self.model(input_ids=tid, token_type_ids=ttid, attention_mask=mask).logits.argmax(dim=1)

            gpu_memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
            total_gpu_memory_usage += gpu_memory_usage

            # 使用nvidia-smi获取GPU利用率
            cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
            gpu_utilization = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            gpu_utilization = [int(value) for value in gpu_utilization.strip().split('\n')]
            total_gpu_utilization += sum(gpu_utilization)

        # 计算平均值
        num_batches = len(self.dataloader)
        avg_gpu_memory_usage = total_gpu_memory_usage / num_batches
        avg_gpu_utilization = total_gpu_utilization / (num_batches * len(gpu_utilization))
        end_time = time.time()

        return {
            'avg_gpu_memory_usage_MB': avg_gpu_memory_usage,
            'avg_gpu_utilization_percent': avg_gpu_utilization,
            'total_time_seconds': end_time - start_time
        }

    # 测量硬件占用率
    def occupancy_rate(self):
        """
            @description:打印函数，打印硬件使用情况说明
            @param None
            @return: None
        """
        results = self.measure_gpu_usage()
        print("平均GPU利用内存（MB）:", round(results['avg_gpu_memory_usage_MB'], 2))
        print("平均GPU内存占用率（%）:", round(results['avg_gpu_utilization_percent'], 2))
        print("总时间（秒）:", round(results['total_time_seconds'], 2))


