import numpy as np
import os
import argparse
import sys

def compare_npy_files(path1, path2, tol=1e-5):
    """
    比较两个 .npy 文件是否相等
    :param path1: 第一个文件路径
    :param path2: 第二个文件路径
    :param tol: 浮点数比较的绝对容差 (默认 1e-5)
    :return: bool, 是否相等
    """
    # 1. 路径检查
    for p in [path1, path2]:
        if not os.path.exists(p):
            print(f"❌ 错误：文件不存在 -> {p}")
            return False

    # 2. 加载数据
    try:
        arr1 = np.load(path1)
        arr2 = np.load(path2)
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return False

    print(f"📦 文件形状: {arr1.shape} | 数据类型: {arr1.dtype} -> {arr2.dtype}")

    # 3. 形状检查
    if arr1.shape != arr2.shape:
        print(f"❌ 形状不匹配: {arr1.shape} != {arr2.shape}")
        return False

    # 4. 精确匹配 (支持 NaN)
    try:
        exact = np.array_equal(arr1, arr2, equal_nan=True)
    except TypeError:
        exact = np.array_equal(arr1, arr2)  

    if exact:
        print("✅ 完全一致 (Exact Match)")
        return True

    # 5. 浮点数近似匹配
    if np.issubdtype(arr1.dtype, np.floating) or np.issubdtype(arr2.dtype, np.float64):
        # 统一转为 float64 计算误差，防止溢出
        diff = np.abs(arr1.astype(np.float64) - arr2.astype(np.float64))
        max_err = np.max(diff)
        is_close = np.allclose(arr1, arr2, rtol=tol, atol=tol)

        if is_close:
            print(f"✅ 数值一致 (近似匹配, 容差={tol})")
            print(f"   📉 最大绝对误差: {max_err:.6e}")
            return True
        else:
            mismatch_ratio = np.mean(diff > tol) * 100
            print(f"❌ 数值不一致")
            print(f"   📉 最大绝对误差: {max_err:.6e} (超出容差 {tol})")
            print(f"   📊 不匹配元素比例: {mismatch_ratio:.4f}%")
            return False
    else:
        print("❌ 不一致 (非浮点类型且未精确相等)")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较两个 .npy 文件是否相等")
    parser.add_argument("file1", type=str, help="第一个 .npy 文件路径")
    parser.add_argument("file2", type=str, help="第二个 .npy 文件路径")
    parser.add_argument("--tol", type=float, default=1e-5, help="浮点数比较容差 (默认 1e-5)")
    args = parser.parse_args()

    compare_npy_files(args.file1, args.file2, args.tol)