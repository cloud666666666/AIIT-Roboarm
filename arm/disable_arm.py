"""回归初始位置并失能机械臂。"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_base import Arm

if __name__ == "__main__":
    arm = Arm()
    arm.disconnect_arm()
