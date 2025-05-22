import functools

import numpy as np
import pyrender
import rlemasklib
import trimesh
from nlf_pipeline.util.paths import DATA_ROOT

def render_rle(vertices, camera, imshape, n_verts_subset):
    return get_render_fn(n_verts_subset)(vertices, camera, imshape)


@functools.lru_cache()
def get_render_fn(n_verts_subset):
    if n_verts_subset == 6890:
        i_verts = np.arange(6890)
        faces = np.load(f'{DATA_ROOT}/nlf/smpl_faces.npy')
    else:
        vertex_subset = np.load(f'{DATA_ROOT}/body_models/smpl/vertex_subset_{n_verts_subset}.npz')
        i_verts = vertex_subset['i_verts']
        faces = vertex_subset['faces']

    scene = pyrender.Scene()
    pyrender_camera = CustomIntrinsicsCamera()
    camera_node = scene.add(pyrender_camera, name='pc-camera')
    renderer = pyrender.OffscreenRenderer(10, 10)

    def render_batch(vertices_batch, camera, imshape):
        if len(vertices_batch) == 0:
            return np.empty((0, imshape[0], imshape[1]), dtype=np.uint8)

        renderer.viewport_height = imshape[0]
        renderer.viewport_width = imshape[1]

        pyrender_camera.intrinsic_matrix = camera.intrinsic_matrix
        extr = camera.get_extrinsic_matrix()
        extr[:3, 3] /= 1000
        extr[1:3] *= -1
        scene.set_pose(camera_node, pose=np.linalg.inv(extr))

        results = []
        for vertices in vertices_batch:
            verts_transformed = vertices[i_verts] / 1000
            node = scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts_transformed, faces)))
            depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)
            scene.remove_node(node)
            results.append(rlemasklib.RLEMask.from_array(depth != 0))
        return results

    return render_batch

class CustomIntrinsicsCamera(pyrender.camera.Camera):
    def __init__(self, intrinsic_matrix=None):
        super().__init__()
        self.intrinsic_matrix = intrinsic_matrix

    def get_projection_matrix(self, width, height):
        P = np.zeros((4, 4), np.float32)
        P[0, 0] = 2 * self.intrinsic_matrix[0, 0] / width
        P[1, 1] = 2 * self.intrinsic_matrix[1, 1] / height

        P[0, 2] = 1 - 2 * (self.intrinsic_matrix[0, 2] + 0.5) / width
        P[1, 2] = 2 * (self.intrinsic_matrix[1, 2] + 0.5) / height - 1
        P[3, 2] = -1
        n = 0.05
        f = 100
        P[2, 2] = (f + n) / (n - f)
        P[2, 3] = (2 * f * n) / (n - f)
        return P
