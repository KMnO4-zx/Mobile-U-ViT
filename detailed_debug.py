import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from network.MobileUViT import MobileUViT

def visualize_feature_maps():
    """可视化特征图的变化"""
    
    # 创建模型和输入
    model = MobileUViT()
    model.eval()
    
    # 创建示例输入（医学图像风格）
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    
    # 收集各层特征图
    feature_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                feature_maps[name] = [o.detach() for o in output]
            else:
                feature_maps[name] = output.detach()
        return hook
    
    # 注册hook
    hooks = []
    hooks.append(model.patch_embeddings.register_forward_hook(hook_fn('embeddings')))
    hooks.append(model.down.register_forward_hook(hook_fn('downsample')))
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    # 清理hooks
    for hook in hooks:
        hook.remove()
    
    return model, input_tensor, feature_maps

def analyze_parameter_efficiency():
    """分析参数效率"""
    model = MobileUViT()
    
    param_analysis = {}
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                param_analysis[name] = params
                total_params += params
    
    # 按模块类型统计
    module_types = {
        'Conv2d': 0,
        'Linear': 0,
        'BatchNorm2d': 0,
        'LayerNorm': 0,
        'Others': 0
    }
    
    for name, params in param_analysis.items():
        if 'Conv2d' in name:
            module_types['Conv2d'] += params
        elif 'Linear' in name:
            module_types['Linear'] += params
        elif 'BatchNorm2d' in name:
            module_types['BatchNorm2d'] += params
        elif 'LayerNorm' in name:
            module_types['LayerNorm'] += params
        else:
            module_types['Others'] += params
    
    return param_analysis, module_types, total_params

def detailed_module_analysis():
    """详细模块分析"""
    
    print("=== 详细模块功能分析 ===\n")
    
    print("📊 1. ConvUtr 深度解析:")
    print("   ├─ 深度可分离卷积: 3x3分组卷积 + 1x1点卷积")
    print("   ├─ 残差连接: 保持梯度流动")
    print("   ├─ GELU激活: 平滑非线性")
    print("   └─ 批归一化: 稳定训练")
    
    print("\n🔍 2. Embeddings 模块架构:")
    print("   ├─ Stem: 3→16通道的基础特征提取")
    print("   ├─ Layer1: 16→16通道，核大小3，深度1")
    print("   ├─ Layer2: 16→32通道，核大小3，深度1") 
    print("   ├─ Layer3: 32→64通道，核大小7，深度3")
    print("   └─ 输出: 1/8分辨率的64通道特征图")
    
    print("\n⚡ 3. LGLBlock 机制:")
    print("   ├─ LocalAgg: 9x9大核卷积进行局部聚合")
    print("   ├─ 门控机制: Sigmoid控制信息流")
    print("   ├─ GlobalSparseAttn: 稀疏全局注意力")
    print("   ├─ SR-ratio=2: 空间降采样减少计算")
    print("   └─ 残差连接: 保持原始信息")
    
    print("\n🌍 4. GlobalSparseAttn 原理:")
    print("   ├─ QKV投影: 64维→192维")
    print("   ├─ 多头注意力: 8个头，每头8维")
    print("   ├─ 空间降采样: 2倍减少计算量")
    print("   ├─ 转置卷积上采样: 恢复空间分辨率")
    print("   └─ LayerNorm: 稳定注意力计算")
    
    print("\n🔧 5. 解码器设计:")
    print("   ├─ 跳跃连接: 融合多尺度特征")
    print("   ├─ 逐步上采样: 2倍上采样×3次")
    print("   ├─ 通道压缩: 64→32→16→1")
    print("   └─ 1x1卷积: 最终分割预测")

def memory_analysis():
    """内存使用分析"""
    model = MobileUViT()
    
    # 分析不同输入尺寸的内存使用
    input_sizes = [(1, 3, 128, 128), (1, 3, 256, 256), (1, 3, 512, 512)]
    
    memory_usage = {}
    
    for size in input_sizes:
        input_tensor = torch.randn(*size)
        
        # 计算激活内存（近似）
        def get_memory_usage():
            model.eval()
            with torch.no_grad():
                # 使用hook收集特征图大小
                feature_sizes = []
                
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        feature_sizes.append(output.numel())
                    elif isinstance(output, tuple):
                        feature_sizes.extend([o.numel() for o in output if isinstance(o, torch.Tensor)])
                
                hooks = []
                for module in model.modules():
                    if len(list(module.children())) == 0:
                        hooks.append(module.register_forward_hook(hook_fn))
                
                _ = model(input_tensor)
                
                for hook in hooks:
                    hook.remove()
                
                return sum(feature_sizes) * 4 / (1024**2)  # MB
        
        memory_usage[f"{size[-2]}x{size[-1]}"] = get_memory_usage()
    
    return memory_usage

def create_architecture_summary():
    """创建架构总结"""
    
    summary = """
Mobile-UViT 架构总结
==================

🎯 设计目标：移动设备上的轻量级医学图像分割

📐 输入输出：
   - 输入: RGB图像 [B,3,256,256] 或灰度图转3通道
   - 输出: 分割掩码 [B,1,256,256]

🏗️ 核心架构：
   1. 编码器：多尺度特征提取（ConvUtr块）
   2. 瓶颈：LGL（局部-全局-局部）注意力机制
   3. 解码器：跳跃连接+逐步上采样

💡 创新点：
   - ConvUtr：参数高效的卷积模块
   - LGL机制：局部全局特征融合
   - 稀疏注意力：降低计算复杂度
   - 移动优先：1.39M参数，适合移动设备

⚙️ 配置参数：
   ├─ 基础版: dims=[16,32,64,128], 1.39M参数
   ├─ 大版: dims=[32,64,128,256], 5.5M参数
   └─ 深度: depths=[1,1,3,3,3]

🔍 关键特性：
   - 感受野：从3x3到7x7的大核卷积
   - 注意力：8头注意力，64维嵌入
   - 正则化：DropPath + BatchNorm + LayerNorm
   - 激活：GELU + ReLU组合
    """
    
    return summary

if __name__ == "__main__":
    # 运行详细分析
    detailed_module_analysis()
    
    # 参数效率分析
    param_analysis, module_types, total_params = analyze_parameter_efficiency()
    
    print("\n" + "="*50)
    print("📊 参数效率分析")
    print("="*50)
    print(f"总参数量: {total_params:,}")
    print("\n按模块类型分布:")
    for module_type, params in module_types.items():
        percentage = (params / total_params) * 100
        print(f"  {module_type}: {params:,} ({percentage:.1f}%)")
    
    # 内存使用分析
    memory_usage = memory_analysis()
    print("\n" + "="*50)
    print("💾 内存使用分析")
    print("="*50)
    for size, memory in memory_usage.items():
        print(f"输入 {size}: {memory:.2f} MB")
    
    # 打印架构总结
    print("\n" + create_architecture_summary())
    
    # 验证模型可以运行
    print("\n" + "="*50)
    print("✅ 模型验证")
    print("="*50)
    
    model = MobileUViT()
    model.eval()
    
    with torch.no_grad():
        # 测试不同输入
        test_inputs = [
            torch.randn(1, 3, 256, 256),  # 标准输入
            torch.randn(1, 1, 256, 256),  # 灰度图输入
        ]
        
        for i, test_input in enumerate(test_inputs):
            if test_input.shape[1] == 1:
                print(f"灰度图输入 {test_input.shape} -> 自动转换为3通道")
            
            output = model(test_input)
            print(f"测试 {i+1}: {test_input.shape} -> {output.shape} ✓")