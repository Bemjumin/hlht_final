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

        for inputs, _ in self.dataloader:
            inputs = inputs.to(self.device)

            # 测量GPU内存占用率
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                outputs = self.model(inputs)
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

    def computation(self):
        """
            @description:进行性能测试二：衡量算法/模型的复杂度 (FLOPS)
            @param None
            @return: float: (FLOPS大小)
        """
        if self.model is None:
            raise ValueError("Model not set. Use set_model() to set the model_bert.")
            # 将输入数据传递给模型以获取中间的各层信息
        for data in self.dataloader:
            input_tensor, _ = data
            break
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            start = time.time()
            _ = self.model(input_tensor)
            end = time.time()

        flops = 0
        input_size = input_tensor.size()[2:]
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                # 计算卷积层的FLOPs
                output_size = module.weight.size()[2:]
                flops += (2 * module.in_channels * module.out_channels * module.kernel_size[0] *
                          module.kernel_size[1] * output_size[0] * output_size[1])
                # /(module.stride[0] * module.stride[1]))
            elif isinstance(module, torch.nn.Linear):
                # 计算全连接层的FLOPs
                flops += (2 * module.in_features * module.out_features)
        print('the FLOPs/run time is: ', round((flops / 1024000) / (end - start), 2), 'kM/s')
        param_nums = 0
        for param in self.model.parameters():
            param_nums += param.numel()
        print(f"Total parameters: {param_nums}")

        return round((flops / 1000000) / (end - start), 2)

    def compute_sparse_degree(self):
        """
            @description:计算网络的稀疏度
            @param: None
            @return: None
        """
        # 定义卷积层的参数量计算函数
        def count_conv_params(m):
            if isinstance(m, torch.nn.Conv2d):
                return m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]

        # 定义全连接层的参数量计算函数
        def count_fc_params(m):
            if isinstance(m, torch.nn.Linear):
                return m.in_features * m.out_features
        # 计算卷积层和全连接层的参数量
        conv_params = sum(count_conv_params(m) for m in self.model.modules() if count_conv_params(m) is not None)
        fc_params = sum(count_fc_params(m) for m in self.model.modules() if count_fc_params(m) is not None)

        # 计算总参数量
        total_params = sum(p.numel() for p in self.model.parameters())

        # 计算稀疏度
        sparse_degree = (conv_params + fc_params) / total_params
        print('sparse_degree：', round(sparse_degree, 5) * 100, '%')

        return sparse_degree
