import numpy as np
import os

SPLAT_GAUSSIAN_BYTES = 32


def fix_splat_orientation_bytes(splat_data):
    # .splat 파일은 가우시안당 32바이트 구조입니다.
    # 구조: pos(3*float32), scale(3*float32), color(4*uint8), rot(4*uint8)

    data = np.frombuffer(splat_data, dtype=np.uint8).copy()
    if len(data) % SPLAT_GAUSSIAN_BYTES != 0:
        raise ValueError(
            f".splat data size must be a multiple of {SPLAT_GAUSSIAN_BYTES} bytes, got {len(data)} bytes."
        )

    num_gaussians = len(data) // SPLAT_GAUSSIAN_BYTES
    
    # 2D 배열로 재구성 (N x 32)
    gaussians = data.reshape(num_gaussians, SPLAT_GAUSSIAN_BYTES)
    
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
    
    return gaussians.tobytes()


def fix_splat_orientation(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"파일을 찾을 수 없습니다: {input_file}")
        return

    with open(input_file, "rb") as f:
        splat_data = f.read()

    fixed_splat_data = fix_splat_orientation_bytes(splat_data)

    with open(output_file, "wb") as f:
        f.write(fixed_splat_data)
    print(f"회전 수정 완료: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply a 180-degree X-axis conversion to a .splat file.")
    parser.add_argument("input_path", help="Input .splat path")
    parser.add_argument("output_path", help="Output .splat path")
    args = parser.parse_args()

    fix_splat_orientation(args.input_path, args.output_path)
