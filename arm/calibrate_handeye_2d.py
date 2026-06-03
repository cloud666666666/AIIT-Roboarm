# Description: 机械臂手眼标定2D版
# 使用方法：准备一个显眼的点（如一个小球），用鼠标点击图片上该点的位置
# 再将机械臂末端移动到该位置（要求夹爪static连杆垂直于桌面，即保证夹爪根部和末端xy坐标相同），按空格键记录图片点与末端位姿
# 改变点的位置，重复4次以上，越多误差越小，按ESC键退出计算标定结果
# 标定完成后会得到一个矩阵，表示相机坐标系（二维）到机械臂基座坐标系（z轴为桌面不变，故也是二维）的变换
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera
from utils.config_getter import get_config_value
from utils.cv2_display import (
    show_image,
    poll_key,
    set_mouse_callback,
    destroy_all_windows,
)
from arm.arm_base import Arm
import argparse
import cv2
import numpy as np
import time
import threading


TEACHING_POINT_LABELS = tuple(str(i + 1) for i in range(16))


def pack_end_pose(
    end_pos: list[float] | None, end_rot_deg_zyx: list[float] | None
) -> np.ndarray | None:
    if end_pos is None or end_rot_deg_zyx is None:
        return None
    return np.array([*end_pos, *end_rot_deg_zyx], dtype=np.float32)


def image_to_robot_xy(
    homography_matrix: np.ndarray, image_point: tuple[float, float] | np.ndarray
) -> np.ndarray:
    image_point_homogeneous = np.array(
        [image_point[0], image_point[1], 1.0], dtype=np.float64
    )
    robot_point_homogeneous = homography_matrix @ image_point_homogeneous
    return robot_point_homogeneous[:2] / robot_point_homogeneous[2]


def detect_chessboard_points(
    color_image: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found or corners is None:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined_corners.reshape(-1, 2).astype(np.float32)


def index_to_row_col(index: int, pattern_size: tuple[int, int]) -> tuple[int, int]:
    cols, _ = pattern_size
    row = index // cols
    col = index % cols
    return row, col


def select_teaching_corner_indices(pattern_size: tuple[int, int]) -> np.ndarray:
    cols, rows = pattern_size
    if cols < 2 or rows < 2:
        raise ValueError("棋盘格内角点行列数至少都要大于等于2")

    top_row = 0
    bottom_row = rows - 1
    left_col = 0
    right_col = cols - 1
    mid_row = rows // 2
    mid_col = cols // 2

    ordered_points = [
        (top_row, left_col),
        (top_row, mid_col),
        (top_row, right_col),
        (mid_row, right_col),
        (bottom_row, right_col),
        (bottom_row, mid_col),
        (bottom_row, left_col),
        (mid_row, left_col),
        (mid_row, mid_col),
    ]

    unique_points: list[tuple[int, int]] = []
    for point in ordered_points:
        if point not in unique_points:
            unique_points.append(point)

    return np.array([row * cols + col for row, col in unique_points], dtype=np.int32)


def make_board_index_points(pattern_size: tuple[int, int]) -> np.ndarray:
    cols, rows = pattern_size
    points = []
    for row in range(rows):
        for col in range(cols):
            points.append([float(col), float(row)])
    return np.array(points, dtype=np.float32)


def draw_reference_points(
    color_image: np.ndarray,
    corners: np.ndarray,
    reference_indices: np.ndarray,
    active_reference_idx: int | None = None,
) -> np.ndarray:
    image_to_show = color_image.copy()
    for x, y in corners:
        cv2.circle(image_to_show, (int(x), int(y)), 4, (0, 255, 0), -1)
    for label_idx, corner_index in enumerate(reference_indices):
        x, y = corners[int(corner_index)]
        color = (0, 255, 255) if label_idx == active_reference_idx else (0, 0, 255)
        cv2.circle(image_to_show, (int(x), int(y)), 10, color, 2)
        label = TEACHING_POINT_LABELS[label_idx]
        cv2.putText(
            image_to_show,
            label,
            (int(x) + 12, int(y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
    return image_to_show


def calibrate_2d_from_correspondences(
    image_points: np.ndarray, robot_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    image_points = np.asarray(image_points, dtype=np.float32)
    robot_points = np.asarray(robot_points, dtype=np.float32)

    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("图片点数据格式错误，应该是Nx2")
    if robot_points.ndim != 2 or robot_points.shape[1] != 2:
        raise ValueError("机械臂平面点数据格式错误，应该是Nx2")
    if image_points.shape[0] != robot_points.shape[0]:
        raise ValueError("图片点和机械臂平面点数量不匹配")
    if image_points.shape[0] < 4:
        raise ValueError("标定点数量不足，至少需要4个点")

    homography_matrix, inlier_mask = cv2.findHomography(
        image_points, robot_points, cv2.RANSAC, 5.0
    )
    if homography_matrix is None or inlier_mask is None:
        raise RuntimeError("计算单应性矩阵失败")
    return homography_matrix.astype(np.float32), inlier_mask.reshape(-1).astype(bool)


def compute_reprojection_errors(
    homography_matrix: np.ndarray, image_points: np.ndarray, robot_points: np.ndarray
) -> np.ndarray:
    projected_points = np.array(
        [image_to_robot_xy(homography_matrix, point) for point in image_points],
        dtype=np.float32,
    )
    return np.linalg.norm(projected_points - robot_points, axis=1)


# 收集图片和机械臂末端坐标数据
def collect_image_pose(image_points_path, end_poses_path):
    image_points: list[tuple[int, int]] = []
    end_poses: list[np.ndarray] = []
    selected_point: tuple[int, int] | None = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            print(f"选择图片点: ({x}, {y})，将机械臂移动到该点后按空格记录。")

    window_name = "Camera"
    set_mouse_callback(window_name, mouse_callback)

    arm = Arm()
    arm.disable_torque()
    cam = Camera(color=True, depth=False)
    while True:
        try:
            frames = cam.get_frames()
            color_image = frames.get("color")
            if color_image is None:
                print("failed to get color image")
                continue

            image_to_show = color_image.copy()
            if selected_point is not None:
                cv2.circle(image_to_show, selected_point, 5, (0, 0, 255), -1)
            cv2.putText(
                image_to_show,
                f"pairs: {len(image_points)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            show_image(window_name, image_to_show)

            key = poll_key(1)
            if key == 27:
                break
            if key == ord(" "):
                if selected_point is None:
                    print("请先点击图片点，再按空格记录。")
                    continue

                end_pos, end_rot_deg_zyx = arm.get_arm_pose()
                end_pose = pack_end_pose(end_pos, end_rot_deg_zyx)
                if end_pose is None:
                    print("获取机械臂末端位姿失败")
                    continue

                image_points.append(selected_point)
                end_poses.append(end_pose)
                print("记录图片点:", selected_point)
                print(
                    "记录末端位姿 [x, y, z, RZ, RY, RX] (欧拉角顺序为ZYX):",
                    end_pose.tolist(),
                )
                selected_point = None
        except KeyboardInterrupt:
            break
    destroy_all_windows()

    np.save(image_points_path, np.array(image_points, dtype=np.float32))
    np.save(end_poses_path, np.array(end_poses, dtype=np.float32))

    return image_points_path, end_poses_path


def teach_board_reference_points(
    arm: Arm,
    cam: Camera,
    pattern_size: tuple[int, int],
    reference_indices: np.ndarray,
    round_index: int,
    capture_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    window_name = "Camera"
    locked_corners: np.ndarray | None = None
    robot_points: list[np.ndarray] = []
    current_step = 0
    print(
        f"请先固定棋盘格。第 {round_index + 1}/{capture_count} 轮：按回车锁定当前棋盘位置，随后按提示依次示教各个角点。"
    )

    while current_step < len(reference_indices):
        frames = cam.get_frames()
        color_image = frames.get("color")
        if color_image is None:
            print("failed to get color image")
            continue

        corners = detect_chessboard_points(color_image, pattern_size)
        image_to_show = color_image.copy()

        if locked_corners is None:
            if corners is not None:
                image_to_show = draw_reference_points(
                    color_image, corners, reference_indices, None
                )
            cv2.putText(
                image_to_show,
                "Press Enter to lock current chessboard",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_to_show,
                "Need full chessboard visible before locking",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            show_image(window_name, image_to_show)

            key = poll_key(1)
            if key == 27:
                raise KeyboardInterrupt
            if key != 13:
                continue
            if corners is None:
                print("当前未检测到完整棋盘格，无法锁定。")
                continue

            locked_corners = corners.copy()
            print("已锁定棋盘，接下来按提示依次示教各个角点。")
            continue

        image_to_show = draw_reference_points(
            color_image, locked_corners, reference_indices, current_step
        )
        current_corner_index = int(reference_indices[current_step])
        row, col = index_to_row_col(current_corner_index, pattern_size)
        cv2.putText(
            image_to_show,
            f"Move gripper to point {current_step + 1} then press Space",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            image_to_show,
            f"Target row={row}, col={col}. Enter unlocks chessboard.",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        show_image(window_name, image_to_show)

        key = poll_key(1)
        if key == 27:
            raise KeyboardInterrupt
        if key == 13:
            locked_corners = None
            current_step = 0
            robot_points.clear()
            print("已解锁棋盘，请重新调整棋盘并按回车锁定。")
            continue
        if key != ord(" "):
            continue

        end_pos, end_rot_deg_zyx = arm.get_arm_pose()
        end_pose = pack_end_pose(end_pos, end_rot_deg_zyx)
        if end_pose is None:
            print("获取机械臂末端位姿失败")
            continue

        robot_points.append(end_pose[:2].astype(np.float32))
        print(
            f"已记录点 {current_step + 1} (row={row}, col={col}) -> "
            f"({end_pose[0]:.6f}, {end_pose[1]:.6f})"
        )
        current_step += 1

    assert locked_corners is not None
    image_points = locked_corners[reference_indices].astype(np.float32)
    return image_points, np.array(robot_points, dtype=np.float32)


def collect_board_correspondences(
    image_points_path: str,
    robot_points_path: str,
    reference_indices_path: str,
    reference_robot_points_path: str,
    pattern_size: tuple[int, int],
    capture_count: int,
):
    window_name = "Camera"

    arm = Arm()
    arm.disable_torque()
    cam = Camera(color=True, depth=False)

    try:
        reference_indices = select_teaching_corner_indices(pattern_size)
        all_image_points: list[np.ndarray] = []
        all_robot_points: list[np.ndarray] = []

        for round_index in range(capture_count):
            image_points, robot_points = teach_board_reference_points(
                arm,
                cam,
                pattern_size,
                reference_indices,
                round_index,
                capture_count,
            )
            all_image_points.append(image_points)
            all_robot_points.append(robot_points)

        merged_image_points = np.concatenate(all_image_points, axis=0).astype(
            np.float32
        )
        merged_robot_points = np.concatenate(all_robot_points, axis=0).astype(
            np.float32
        )

        homography_matrix, inlier_mask = calibrate_2d_from_correspondences(
            merged_image_points, merged_robot_points
        )
        np.save(image_points_path, merged_image_points)
        np.save(robot_points_path, merged_robot_points)
        np.save(reference_indices_path, reference_indices)
        np.save(reference_robot_points_path, merged_robot_points)
        return image_points_path, robot_points_path, homography_matrix, inlier_mask
    finally:
        destroy_all_windows()
        cam.close()
        arm.disconnect_arm()


def calibrate_2d(image_points_path, end_poses_path):
    image_points = np.load(image_points_path).astype(np.float32)
    end_poses = np.load(end_poses_path).astype(np.float32)

    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("图片点数据格式错误，应该是Nx2")
    if end_poses.ndim != 2 or end_poses.shape[1] < 2:
        raise ValueError("机械臂末端位姿数据格式错误，应该至少包含x和y坐标")

    print("机械臂末端位姿 [x, y, z, RZ, RY, RX] (欧拉角顺序为ZYX):")
    print(end_poses)

    homography_matrix, _ = calibrate_2d_from_correspondences(
        image_points, end_poses[:, :2]
    )
    return homography_matrix


def test_homography(homography_matrix, image_point):
    arm = Arm()
    arm.move_to_home()
    time.sleep(1)

    robot_point = image_to_robot_xy(homography_matrix, image_point)
    z = get_config_value("default_desktop_height")
    print(f"测试点 {image_point} 对应机械臂末端位置 {robot_point.tolist()}")
    arm.move_to(
        [float(robot_point[0]), float(robot_point[1]), z],
        gripper_open_0to1=1,
        rot_rad=0,
    )
    time.sleep(1)

    arm.move_to_home()
    time.sleep(1)
    arm.disconnect_arm()


def test_moveto(arm: Arm, homography_matrix, image_point, move_lock: threading.Lock):
    with move_lock:
        arm.move_to_home()
        time.sleep(0.1)

        target_x, target_y = image_to_robot_xy(homography_matrix, image_point)
        target_z = get_config_value("default_desktop_height")
        print(
            f"Clicked image point: ({image_point[0]}, {image_point[1]}), "
            f"Mapped arm position: ({target_x}, {target_y})"
        )
        arm.move_to(
            [float(target_x), float(target_y), target_z],
            gripper_open_0to1=1,
            rot_rad=0,
        )


def test_handeye_2d(homography_matrix):
    arm = Arm()
    move_lock = threading.Lock()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Left button clicked at ({x}, {y})")
            threading.Thread(
                target=test_moveto,
                args=(arm, homography_matrix, (x, y), move_lock),
                daemon=True,
            ).start()

    window_name = "Camera"
    set_mouse_callback(window_name, mouse_callback)

    cam = Camera(color=True, depth=False)
    while True:
        try:
            frames = cam.get_frames()
            color_image = frames.get("color")
            if color_image is None:
                print("failed to get color image")
                time.sleep(0.5)
                continue
            show_image(window_name, color_image)

            key = poll_key(1)
            if key == 27:
                break

        except KeyboardInterrupt:
            break

    destroy_all_windows()
    cam.close()

    arm.disconnect_arm()


def main():
    argparser = argparse.ArgumentParser(description="机械臂手眼标定2D版")
    argparser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="模式: calibrate 手动采集多点; calibrate_board 半自动采集多点; test 测试",
    )
    argparser.add_argument(
        "--pattern-cols", type=int, default=7, help="棋盘格每行内角点数"
    )
    argparser.add_argument(
        "--pattern-rows", type=int, default=7, help="棋盘格每列内角点数"
    )
    argparser.add_argument(
        "--capture-count",
        type=int,
        default=1,
        help="calibrate_board模式下重复示教的轮数，多轮数据会拼接后共同拟合单应性矩阵",
    )
    args = argparser.parse_args()

    arm_backend = get_config_value("arm_backend", "real", raise_if_missing=False)
    if arm_backend == "sim" and args.mode in ("calibrate", "calibrate_board"):
        print(
            "手眼标定依赖真实相机与人工拖动机械臂，仿真模式下暂不支持采集，已退出"
        )
        return

    data_dir = os.path.join(os.path.dirname(__file__), "hand-eye-data")
    image_points_path = os.path.join(data_dir, "2d_image_points.npy")
    end_poses_path = os.path.join(data_dir, "2d_end_poses.npy")
    robot_points_path = os.path.join(data_dir, "2d_robot_points.npy")
    reference_indices_path = os.path.join(data_dir, "2d_reference_corner_indices.npy")
    reference_robot_points_path = os.path.join(
        data_dir, "2d_reference_robot_points.npy"
    )
    homography_matrix_path = os.path.join(data_dir, "2d_homography.npy")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if args.mode == "calibrate":
        collect_image_pose(image_points_path, end_poses_path)
        homography_matrix = calibrate_2d(image_points_path, end_poses_path)
        np.save(homography_matrix_path, homography_matrix)

        end_poses = np.load(end_poses_path).astype(np.float32)
        print("机械臂末端位姿 [x, y, z, RZ, RY, RX] (欧拉角顺序为ZYX):")
        print(end_poses)
        points = np.load(image_points_path).astype(np.float32)
        print("图片上点的像素坐标:")
        print(points)
        homography_matrix = np.load(homography_matrix_path)
        print("计算得到的单应性矩阵:")
        print(homography_matrix)
        test_handeye_2d(homography_matrix)
        # for point in points:
        #     test_homography(homography_matrix, point)
    elif args.mode == "calibrate_board":
        pattern_size = (args.pattern_cols, args.pattern_rows)
        _, _, homography_matrix, inlier_mask = collect_board_correspondences(
            image_points_path,
            robot_points_path,
            reference_indices_path,
            reference_robot_points_path,
            pattern_size,
            args.capture_count,
        )
        np.save(homography_matrix_path, homography_matrix)

        image_points = np.load(image_points_path).astype(np.float32)
        robot_points = np.load(robot_points_path).astype(np.float32)
        errors = compute_reprojection_errors(
            homography_matrix, image_points, robot_points
        )
        inlier_errors = errors[inlier_mask]
        print(f"总角点数: {len(image_points)}")
        print(f"RANSAC内点数: {int(inlier_mask.sum())}")
        print(f"全部点平均误差: {float(errors.mean()):.6f} m")
        print(f"全部点最大误差: {float(errors.max()):.6f} m")
        if len(inlier_errors) > 0:
            print(f"内点平均误差: {float(inlier_errors.mean()):.6f} m")
            print(f"内点最大误差: {float(inlier_errors.max()):.6f} m")
        print("计算得到的单应性矩阵:")
        print(homography_matrix)
    elif args.mode == "test":
        homography_matrix = np.load(homography_matrix_path)
        test_handeye_2d(homography_matrix)


if __name__ == "__main__":
    main()
