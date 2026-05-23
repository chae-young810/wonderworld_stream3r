import numpy as np
import os

def fix_splat_orientation(input_file, output_file):
    # .splat 파일은 가우시안당 32바이트 구조입니다.
    # 구조: pos(3*float32), scale(3*float32), color(4*uint8), rot(4*uint8)
    
    if not os.path.exists(input_file):
        print(f"파일을 찾을 수 없습니다: {input_file}")
        return

    # 바이너리 데이터 로드
    data = np.fromfile(input_file, dtype=np.uint8)
    num_gaussians = len(data) // 32
    
    # 2D 배열로 재구성 (N x 32)
    gaussians = data.reshape(num_gaussians, 32)
    
    # 1. 위치(Position) 수정 (앞 12바이트: x, y, z)
    # float32 뷰로 변환하여 좌표 추출
    pos = gaussians[:, :12].view(np.float32).reshape(-1, 3)
    
    # X축 기준 180도 회전: (x, y, z) -> (x, -y, -z)
    pos[:, 1] *= -1
    pos[:, 2] *= -1
    
    # 2. 회전(Rotation/Quaternion) 수정 (마지막 4바이트: q0, q1, q2, q3)
    # .splat 파일의 쿼터니언은 보통 0~255 사이의 uint8로 양자화되어 있습니다.
    # q_float = (q_uint8 - 128) / 128
    quat_u8 = gaussians[:, 28:32].astype(np.float32)
    q = (quat_u8 - 128.0) / 128.0  # [-1.0, 1.0] 범위로 복원
    
    # 쿼터니언 회전 연산 (w, x, y, z 순서 가정)
    # 180도 X축 회전 쿼터니언 [0, 1, 0, 0]과의 곱셈 결과:
    # w_new = -x, x_new = w, y_new = z, z_new = -y
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    new_q = np.zeros_like(q)
    new_q[:, 0] = -x  # new w
    new_q[:, 1] = w   # new x
    new_q[:, 2] = z   # new y
    new_q[:, 3] = -y  # new z
    
    # 다시 uint8 범위(0~255)로 양자화하여 저장
    gaussians[:, 28:32] = np.clip(new_q * 128.0 + 128.0, 0, 255).astype(np.uint8)
    
    # 파일 저장
    gaussians.tofile(output_file)
    print(f"회전 수정 완료: {output_file}")

# --- 실행부 ---
input_path = "/home/sungonce/chae/logs/wonderworld_stream3r/cathedral_1/Gen-23-05_13-55-56/cathedral_1_finished_3dgs.splat"   # 원본 파일명
output_path = "/home/sungonce/chae/logs/wonderworld_stream3r/cathedral_1/Gen-23-05_13-55-56/cathedral_1_finished_3dgs_fixed_axis.splat" # 결과 파일명

fix_splat_orientation(input_path, output_path)