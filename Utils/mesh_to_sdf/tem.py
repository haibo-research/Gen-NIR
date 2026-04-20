import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import numpy as np
import pyrender
from Utils import image


IMAGE = "s1tt1_mask_res.png"
img = Image.open(IMAGE)
#f = rounded_box((0.9, 0.9, 0.05), 0.01)
#f = f - image(img).extrude(1)
f = image(img).extrude(.05)
f.save('out.stl')


mesh = trimesh.load('out.stl')
points, sdf = sample_sdf_near_surface(mesh, number_of_points=50000)
p = np.where(sdf > 0)
n = np.where(sdf < 0)
pos = np.hstack((points[p], np.expand_dims(sdf[p], axis=1)))
neg = np.hstack((points[n], np.expand_dims(sdf[n], axis=1)))
np.savez('out', pos=pos, neg=neg)


data = np.load("out.npz")
data = np.concatenate((data["pos"], data["neg"]), axis=0)

points = data[:, :3]
sdf = data[:, 3]

colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)
scene = pyrender.Scene()
scene.add(cloud)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

