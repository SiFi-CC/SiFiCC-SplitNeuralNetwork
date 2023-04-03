import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from uproot_methods.classes.TVector3 import TVector3


def unit_vec(vec):
    return vec / np.sqrt(np.dot(vec, vec))


def vec_angle(vec1, vec2):
    return np.arccos(np.clip(np.dot(unit_vec(vec1), unit_vec(vec2)), -1.0, 1.0))


def get_phi(vec):
    tvec = TVector3(vec[0], vec[1], vec[2])
    return tvec.phi


def get_theta(vec):
    tvec = TVector3(vec[0], vec[1], vec[2])
    return tvec.theta


# ----------------------------------------------------------------------------------------------------------------------

# plotting
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xlabel("x-axis [mm]")
ax.set_ylabel("y-axis [mm]")
ax.set_zlabel("z-axis [mm]")
ax.plot3D([-10, 10], [0, 0], [0, 0], color="black")
ax.plot3D([0, 0], [10, -10], [0, 0], color="black")
ax.plot3D([0, 0], [0, 0], [10, -10], color="black")

# set the baseline vectors
vec_axis = np.array([0.93781446, 0.23619839, -0.25439018])*10
vec_src = np.array([0.9905789, -0.12525081, -0.05536714])*10
ax.plot3D([0, vec_axis[0]], [0, vec_axis[1]], [0, vec_axis[2]], color="blue", linestyle="--")
ax.plot3D([0, vec_src[0]], [0, vec_src[1]], [0, vec_src[2]], color="green", linestyle="--")

# calculate angle between vectors
angle1 = 2.303
print("Angle: {:.3f} [rad]".format(angle1))
print("Angle: {:.1f}".format(angle1 * 360 / 2 / np.pi))

# build reference vector and rotate it to correct final position
vec_ref = np.array([1, 0, 0])
ax.plot3D([0, vec_ref[0]], [0, vec_ref[1]], [0, vec_ref[2]], color="orange", linestyle="--")

rotation_y = R.from_rotvec((get_theta(vec_axis) - np.pi / 2 - angle1) * np.array([0, 1, 0]))
rotation_z = R.from_rotvec(get_phi(vec_axis) * np.array([0, 0, 1]))
vec_ref = rotation_y.apply(vec_ref)
vec_ref = rotation_z.apply(vec_ref)
ax.plot3D([0, vec_ref[0] * 10], [0, vec_ref[1] * 10], [0, vec_ref[2] * 10], color="orange", linestyle="-")
print("\nAngle: {:.3f} [rad]".format(vec_angle(vec_axis, vec_ref)))

# build cone via axis rotation
list_phi = np.linspace(0, 360, 8)
a = np.sqrt(np.dot(vec_src, vec_src))
for angle in list_phi:
    vec_temp = vec_ref
    rot_vec = np.radians(angle) * vec_axis / np.sqrt(np.dot(vec_axis, vec_axis))
    rot_M = R.from_rotvec(rot_vec)
    vec_temp = rot_M.apply(vec_temp)
    ax.plot3D([0, vec_temp[0] * a], [0, vec_temp[1] * a], [0, vec_temp[2] * a], color="blue")
    print("Temp Angle: {:.3f} [rad]".format(vec_angle(vec_axis, vec_temp)))

plt.show()
