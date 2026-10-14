"""
更好的测试：角速度不沿旋转轴
"""

import numpy as np


def test_angular_velocity_transform():
    """
    测试角速度在不同坐标系间的转换
    """
    print("="*80)
    print("测试：角速度坐标系转换")
    print("="*80)
    
    # 场景1：角速度沿旋转轴（我之前错误的测试）
    print("\n【场景1】角速度沿旋转轴（z轴）")
    print("-"*80)
    
    # 绕z轴旋转30度
    angle = np.pi / 6
    c, s = np.cos(angle), np.sin(angle)
    R_z = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=float)
    
    ang_local = np.array([0.0, 0.0, 0.1])  # 沿z轴
    ang_world = R_z @ ang_local
    
    print(f"旋转矩阵 (绕z轴30°):\n{R_z}")
    print(f"角速度 (local): {ang_local}")
    print(f"角速度 (world): {ang_world}")
    print(f"差异: {np.linalg.norm(ang_world - ang_local):.6f}")
    print("结论: 角速度沿旋转轴时，坐标转换不改变其值 ✓")
    
    # 场景2：角速度不沿旋转轴（正确的测试）
    print("\n【场景2】角速度不沿旋转轴")
    print("-"*80)
    
    ang_local_2 = np.array([0.1, 0.0, 0.0])  # 沿x轴（不是旋转轴）
    ang_world_2 = R_z @ ang_local_2
    
    print(f"角速度 (local): {ang_local_2}")
    print(f"角速度 (world): {ang_world_2}")
    print(f"差异: {np.linalg.norm(ang_world_2 - ang_local_2):.6f}")
    print("结论: 角速度不沿旋转轴时，坐标转换会改变其值 ✓")
    
    # 场景3：实际计算相对速度的影响
    print("\n【场景3】对相对速度计算的影响")
    print("-"*80)
    
    rel_pos_w = np.array([0.1, 0.0, 0.05])
    
    # 使用local frame的角速度（错误）
    cross_local = np.cross(ang_local_2, rel_pos_w)
    print(f"\n使用local frame角速度:")
    print(f"  ω_local = {ang_local_2}")
    print(f"  r_world = {rel_pos_w}")
    print(f"  ω_local × r_world = {cross_local}")
    
    # 使用world frame的角速度（正确）
    cross_world = np.cross(ang_world_2, rel_pos_w)
    print(f"\n使用world frame角速度:")
    print(f"  ω_world = {ang_world_2}")
    print(f"  r_world = {rel_pos_w}")
    print(f"  ω_world × r_world = {cross_world}")
    
    print(f"\n叉乘结果差异: {np.linalg.norm(cross_world - cross_local):.6f}")
    
    if np.linalg.norm(cross_world - cross_local) > 1e-6:
        print("✓ 确认：使用错误的坐标系会导致相对速度计算错误！")
    
    # 场景4：实际机器人的情况
    print("\n【场景4】实际机器人场景模拟")
    print("-"*80)
    print("机器人pelvis绕z轴旋转30°，同时有pitch角速度（绕y轴）")
    
    # Pelvis角速度（body frame）：绕自身y轴旋转
    ang_pelvis_body = np.array([0.0, 0.05, 0.1])  # [roll, pitch, yaw]
    
    # Pelvis的旋转矩阵（world <- body）
    R_pelvis = R_z  # 简化：只考虑yaw
    
    # 转换到world frame
    ang_pelvis_world = R_pelvis @ ang_pelvis_body
    
    print(f"Pelvis角速度 (body):  {ang_pelvis_body}")
    print(f"Pelvis角速度 (world): {ang_pelvis_world}")
    print(f"差异: {np.linalg.norm(ang_pelvis_world - ang_pelvis_body):.6f}")
    
    # Body相对于pelvis的位置（world frame）
    body_pos_rel = np.array([0.2, 0.1, 0.3])  # 例如手臂
    
    # 计算相对速度中的角速度贡献
    cross_body_wrong = np.cross(ang_pelvis_body, body_pos_rel)
    cross_world_correct = np.cross(ang_pelvis_world, body_pos_rel)
    
    print(f"\nBody相对位置 (world): {body_pos_rel}")
    print(f"ω×r (使用body frame ω): {cross_body_wrong}")
    print(f"ω×r (使用world frame ω): {cross_world_correct}")
    print(f"差异: {np.linalg.norm(cross_world_correct - cross_body_wrong):.6f} m/s")
    
    if np.linalg.norm(cross_world_correct - cross_body_wrong) > 0.01:
        print("\n⚠️  差异显著！这会导致body速度计算错误！")
    
    print("\n" + "="*80)
    print("结论:")
    print("="*80)
    print("1. 角速度的坐标系转换很重要")
    print("2. 只有当角速度恰好沿旋转轴时，转换才没有影响")
    print("3. 实际机器人运动中，角速度通常不沿单一轴")
    print("4. 必须确保 ω 和 r 在同一坐标系中才能正确计算 ω×r")
    print("="*80)


if __name__ == "__main__":
    test_angular_velocity_transform()

