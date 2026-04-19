# run.py
import os
import numpy as np
import h5py
from pathlib import Path
import json
from algs import * # 导入所有算法

# ========================
# 1. 实验配置
# ========================
EXPERIMENTS = [
    # 实验 1: ISTA (R=4)
    {
        "name": "ISTA_R4_LowLambda",
        "algorithm": "ISTA",
        "params": {
            "lambda": 1e-6, 
            "iterations": 1000,
            "acceleration": 4  # 每个实验独立配置加速倍率
        }
    },
    # 实验 2: ISTA (R=2) - 验证代码正确性
    {
        "name": "ISTA_R2_Easy",
        "algorithm": "ISTA",
        "params": {
            "lambda": 1e-5, 
            "iterations": 500,
            "acceleration": 2
        }
    },
    # 实验 3: LISTA (R=4)
    {
        "name": "LISTA_R4_Standard",
        "algorithm": "LISTA",
        "params": {
            "T": 10, 
            "acceleration": 4, 
            "model_path": "./checkpoints/lista_model_T10_acc4.pth"
        }
    },
    # 实验 4: LISTA (R=2) - 需要重新训练
    {
        "name": "LISTA_R2_Fast",
        "algorithm": "LISTA",
        "params": {
            "T": 10, 
            "acceleration": 2, 
            "model_path": "./checkpoints/lista_model_T10_acc2.pth"
        }
    }
]

# ========================
# 2. 全局设置与数据加载
# ========================
DATA_DIR = "D:/ProgrammingCoding/7503gp/7503gp-main/7503gp-main/singlecoil_val"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)

def load_raw_data():
    """
    修改点 1: 这个函数现在只负责读取原始数据，不再生成 Mask。
    它返回完整的 K-space 和 Ground Truth，供后续实验按需使用。
    """
    h5_files = sorted(Path(DATA_DIR).glob("*.h5"))
    
    if len(h5_files) < 2:
        raise ValueError("需要至少 2 个 H5 文件 (1个测试, 1个训练)")
        
    print(f"找到 {len(h5_files)} 个文件。")
    test_file = h5_files[0]
    train_files = h5_files[1:] # 后面的都是训练集
    
    # 加载测试数据 (只读一次)
    with h5py.File(test_file, 'r') as f:
        ks = f['kspace'][()]
        slice_idx = ks.shape[0] // 2
        kspace_full = ks[slice_idx]
        
        # 生成真值 (基于全采样数据)
        img_full = np.abs(ifft2c(kspace_full))
        crop_size = min(img_full.shape)
        x_gt = center_crop(img_full, (crop_size, crop_size))
        
        # 归一化 (基于真值)
        scale = x_gt.max()
        x_gt = x_gt / scale if scale > 0 else x_gt
        kspace_full = kspace_full / scale
        
        return {
            "kspace_full": kspace_full, # 返回完整数据
            "ground_truth": x_gt,
            "crop_shape": (crop_size, crop_size)
        }, train_files

# ========================
# 3. 主运行循环
# ========================
def run_experiments():
    # 1. 加载原始数据 (无 Mask)
    raw_data, train_files = load_raw_data()
    print(f"数据加载完成。测试数据形状: {raw_data['ground_truth'].shape}")
    
    results = []
    
    # 2. 循环运行每个实验
    for exp in EXPERIMENTS:
        # --- 核心修改点 2: 在循环内部根据实验配置生成 Mask ---
        current_R = exp['params']['acceleration']
        print(f"\n{'='*60}")
        print(f"运行实验: {exp['name']} (加速倍率 R={current_R})")
        print(f"参数: {exp['params']}")
        
        try:
            # 生成当前实验专用的 Mask
            W = raw_data['kspace_full'].shape[1]
            # 使用固定的 seed 保证不同实验间的 Mask 随机性一致，或者用 exp['name'] 做 seed 保证独立
            mask = create_mask(W, acceleration=current_R, seed=42) 
            
            # 生成欠采样数据 (模拟扫描)
            kspace_u = raw_data['kspace_full'] * mask
            
            # --- 运行 Zero-Filled 基准 (每个 R 都跑一次基准) ---
            x_zf = ifft2c(kspace_u) 
            x_zf_crop = center_crop(np.abs(x_zf), raw_data['crop_shape'])
            psnr_zf = compute_psnr(x_zf_crop, raw_data['ground_truth'])
            ssim_zf = compute_ssim(x_zf_crop, raw_data['ground_truth'])
            print(f"  [基准 R={current_R}] PSNR: {psnr_zf:.2f} dB, SSIM: {ssim_zf:.4f}")
            
            # 保存基准结果
            results.append({
                "experiment_name": f"Zero_Filled_R{current_R}",
                "algorithm": "Zero-Filled",
                "parameters": {"acceleration": current_R},
                "PSNR": psnr_zf,
                "SSIM": ssim_zf
            })

            # --- 运行具体算法 ---
            recon_img = None # 初始化重建图像变量
            
            if exp['algorithm'] == "ISTA":
                x_recon, _ = ista_mri(
                    kspace_u, # 使用当前 R 的欠采样数据
                    mask,
                    lam=exp['params']['lambda'],
                    n_iter=exp['params']['iterations'],
                    x_gt=raw_data['ground_truth'],
                    crop=raw_data['crop_shape']
                )
                recon_img = center_crop(np.abs(x_recon), raw_data['crop_shape'])
                
            elif exp['algorithm'] == "FISTA":
                x_recon, _ = fista_mri(
                    kspace_u,
                    mask,
                    lam=exp['params']['lambda'],
                    n_iter=exp['params']['iterations'],
                    x_gt=raw_data['ground_truth'],
                    crop=raw_data['crop_shape']
                )
                recon_img = center_crop(np.abs(x_recon), raw_data['crop_shape'])
                
            elif exp['algorithm'] == "LISTA":
                model_path = exp['params']['model_path']
                model_accel = exp['params']['acceleration']
                T_layers = exp['params']['T']
                
                # 检查模型路径中的加速倍率是否与实验配置一致 (防止人为错误)
                # 这里只是简单检查，实际文件名可能不包含 acc 信息
                
                if os.path.exists(model_path):
                    print(f"  从 {model_path} 加载预训练模型...")
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = LISTA_MRI(T=T_layers).to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    print(f"  未找到模型。开始训练 LISTA (T={T_layers}, R={model_accel})...")
                    # 训练时也传入 acceleration
                    model, _ = train_lista_model(
                        train_files=train_files,
                        T=T_layers,
                        epochs=50, 
                        acceleration=model_accel, # 确保训练数据也是 R=2 或 R=4
                        save_path=model_path
                    )
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # 推理
                recon_img = lista_inference(
                    model, 
                    kspace_u, # 使用当前 R 的欠采样数据
                    mask.flatten(), 
                    raw_data['crop_shape'],
                    device=device
                )
            else:
                raise ValueError(f"未知算法: {exp['algorithm']}")
            
            # --- 结果评估 ---
            psnr = compute_psnr(recon_img, raw_data['ground_truth'])
            ssim = compute_ssim(recon_img, raw_data['ground_truth'])
            
            result_entry = {
                "experiment_name": exp['name'],
                "algorithm": exp['algorithm'],
                "parameters": exp['params'],
                "PSNR": round(psnr, 4),
                "SSIM": round(ssim, 4)
            }
            results.append(result_entry)
            
            print(f"  [结果] PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
        except Exception as e:
            print(f"  实验失败: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误堆栈
            results.append({
                "experiment_name": exp['name'],
                "error": str(e)
            })
    
    # ========================
    # 4. 输出详细结果
    # ========================
    print(f"\n{'='*80}")
    print("所有实验完成。详细结果如下：")
    print(f"{'='*80}")
    print(f"{'算法':<20} {'参数摘要':<35} {'PSNR (dB)':<12} {'SSIM':<10}")
    print(f"{'-'*80}")
    
    for res in results:
        if 'error' not in res:
            alg_name = res['algorithm']
            params = res['parameters']
            
            # 格式化参数显示
            if alg_name in ["ISTA", "FISTA"]:
                param_str = f"R={params['acceleration']}, λ={params['lambda']}, Iter={params['iterations']}"
            elif alg_name == "LISTA":
                param_str = f"R={params['acceleration']}, T={params['T']}"
            elif alg_name == "Zero-Filled":
                param_str = f"R={params['acceleration']}"
            else:
                param_str = str(params)
                
            print(f"{alg_name:<20} {param_str:<35} {res['PSNR']:<12.4f} {res['SSIM']:<10.4f}")
        else:
            print(f"{res['experiment_name']:<20} {'ERROR':<35} {res['error']}")
    
    # 保存结果到 JSON
    with open(os.path.join(OUTPUT_DIR, "experiment_results.json"), 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"\n详细结果已保存至: {OUTPUT_DIR}/experiment_results.json")

if __name__ == "__main__":
    run_experiments()