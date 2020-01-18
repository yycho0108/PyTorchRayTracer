#!/usr/bin/env python3

import os
import sys
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch as tr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# import kornia.geometry
from raycast import ray_triangle_intersection
import kornia.geometry
import pyqtgraph.opengl as gl
import time

try:
    sys.path.append(os.path.expanduser(
        '~/Repos/Experiments/PhoneBot/control/'))
    from core.vis.viewer.proxy_commands import AddPlotCommand, AddLinesCommand
    from core.vis.viewer.proxy_command import ProxyCommand
    from core.vis.viewer import ProxyViewer
except ImportError:
    print('sad')


class AddGridCommand(ProxyCommand):
    def __init__(self, name='grid', size=(100, 100, 1), spacing=(1, 1, 1)):
        self.size_ = size
        self.spacing_ = spacing
        super().__init__(name)

    def __call__(self, viewer: ProxyViewer):
        item = gl.GLGridItem()
        item.setSize(*self.size_)
        item.setSpacing(*self.spacing_)
        viewer.items_[self.name] = item
        viewer.widget_.addItem(item)


class AddPointsCommand(ProxyCommand):
    def __init__(self, name='points'):
        super().__init__(name)

    def __call__(self, viewer: ProxyViewer):
        item = gl.GLScatterPlotItem()
        item.pos = np.empty((0, 3))  # prevent abort due to pyqtgraph bug
        viewer.items_[self.name] = item
        viewer.handlers_[self.name] = item.setData
        viewer.widget_.addItem(item)


def rotation_matrix_2d(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s], [s, c]])


def rotate_vector(v, axis, origin, angle):
    pass


class Config(object):

    def __init__(self):
        # Ray-tracing configuration.
        self.fov = (0.5*np.deg2rad(45), 2*np.pi)  # Field of view
        self.res = (32, 1024)  # Resolution
        self.fps = 10.0  # 10Hz
        self.win = 0.1  # Aggregation Window, set to 1 rev
        # Scene generation configuration.
        self.xlim = [-60.0, 60.0]
        self.ylim = [-60.0, 60.0]
        self.zlim = [0.0, 3.0]  # TODO(yycho0108): Validate +z=up
        self.lim = np.asarray(
            [self.xlim, self.ylim, self.zlim]).astype(np.float32)  # 3x2
        self.min_num_objects = 1
        self.max_num_objects = 8
        self.max_gen_iterations = 128
        self.dim_lim = np.asarray([
            [0.5, 3.0],
            [1.5, 4.5],
            [1.0, 2.0]], dtype=np.float32)
        self.vmax = 18.0  # m/s
        self.wmax = 0.5  # rad/s
        self.ray_z = 2.0
        self.max_ray_distance = 100.0

        self.finalize()  # Compute cache for derived parameters.

    def finalize(self):
        v_fov, h_fov = self.fov
        v_res, h_res = self.res
        # Create uniformly spaced bins.
        self.v_ang = np.linspace(-v_fov/2, v_fov/2, v_res)
        self.h_ang = np.linspace(-h_fov/2, h_fov/2, h_res)
        # Bins -> Grid
        self.grid = np.stack(np.meshgrid(
            self.v_ang, self.h_ang, indexing='ij'), axis=-1)
        print('gs', self.grid.shape)
        self.v_grid, self.h_grid = [self.grid[..., i] for i in range(2)]
        # Grid -> Rays
        v_cos, v_sin = np.cos(self.v_grid), np.sin(self.v_grid)
        h_cos, h_sin = np.cos(self.h_grid), np.sin(self.h_grid)
        self.rays = np.stack(
            [h_cos * v_cos, h_sin * v_cos, v_sin], axis=-1).astype(np.float32)
        # Build Approximate timestamp offsets (mostly experimental)
        stamps = (1.0 / self.fps) * (self.h_grid / (2*np.pi))
        stamps -= stamps.min()  # start from offset=0, somewhat arbitrarily.
        print('ss', stamps.shape)
        self.stamps = stamps


def get_bounding_box(dim, pose):
    """ NOTE(yycho0108): 2D top-down bbox! """
    ccw_signs = tr.tensor([(1, 1), (1, -1), (-1, -1), (-1, 1)])  # (4,2)
    corners = 0.5 * dim[..., None, :2] * ccw_signs  # (...,4,2)
    c, s = tr.cos(pose[..., -1]), tr.sin(pose[..., -1])
    rmat = tr.stack([c, -s, s, c], dim=-1).reshape(pose.shape[:-1] + (2, 2))
    corners = tr.einsum('...ac,...bc->...ba', rmat, corners)
    corners += pose[..., None, :2]
    return corners


def bbox_intersects(lhs, rhs):
    # assert(lhs.shape == rhs.shape)
    for polygon in [lhs, rhs]:
        # for each polygon, look at each edge of the polygon, and determine if it separates
        # the two shapes
        for i1 in range(len(polygon)):

            # grab 2 vertices to create an edge
            i2 = (i1 + 1) % len(polygon)
            p1 = polygon[i1]
            p2 = polygon[i2]

            # find the line perpendicular to this edge
            nx, ny = p2[1] - p1[1], p1[0] - p2[0]

            minA, maxA = None, None
            # for each vertex in the first shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            for j in range(len(lhs)):
                projected = nx * lhs[j][0] + ny * lhs[j][1]
                if (minA is None) or (projected < minA):
                    minA = projected

                if (maxA is None) or (projected > maxA):
                    maxA = projected

            # for each vertex in the second shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            minB, maxB = None, None
            for j in range(len(rhs)):
                projected = nx * rhs[j][0] + ny * rhs[j][1]
                if (minB is None) or (projected < minB):
                    minB = projected

                if (maxB is None) or (projected > maxB):
                    maxB = projected

            # if there is no overlap between the projects, the edge we are looking at separates the two
            # polygons, and we know there is no overlap
            if (maxA < minB) or (maxB < minA):
                return False
    return True


def create_config() -> Config:
    return Config()


def create_vehicle(num=(), config: Config = Config()):
    # Create distributions.
    dim_dist = tr.distributions.Uniform(
        tr.from_numpy(config.dim_lim[:, 0]).cuda(),
        tr.from_numpy(config.dim_lim[:, 1]).cuda()
        )

    pos_dist = tr.distributions.Uniform(
        tr.tensor([config.xlim[0], config.ylim[0], -np.pi]).cuda(),
        tr.tensor([config.xlim[1], config.ylim[1], np.pi]).cuda()
        )

    vel_dist = tr.distributions.Uniform(
        tr.tensor([0.0, 0.0]).cuda(),
        tr.tensor([config.vmax, config.wmax]).cuda()
    )

    # Generate vehicle from configured distributions.
    dim = dim_dist.rsample(num)
    pos = pos_dist.rsample(num)
    vel = vel_dist.rsample(num)
    v = vel[..., 0]
    w = vel[..., 1]
    h = pos[..., -1]
    c, s = tr.cos(h), tr.sin(h)
    vx, vy = v*c, v*s
    vel = tr.stack([vx, vy, w], dim=-1)
    return (dim, pos), vel


def create_scene(config: Config):
    # Determine number of objects to generate.
    num_objects = np.random.randint(
        config.min_num_objects, config.max_num_objects+1)
    (dim, pos), vel = create_vehicle((num_objects,), config)

    # Also create ground
    gdim = config.lim[..., 1] - config.lim[..., 0]
    gdim[0] += 2.0 * config.max_ray_distance
    gdim[1] += 2.0 * config.max_ray_distance
    gdim[2] = 0.01
    gpos = (0.0, 0.0, 0.0)
    gvel = (0.0, 0.0, 0.0)

    # Append ground.
    dim = tr.cat((dim, tr.tensor(gdim).view(1, 3)), 0)
    pos = tr.cat((pos, tr.tensor(gpos).view(1, 3)), 0)
    vel = tr.cat((vel, tr.tensor(gvel).view(1, 3)), 0)

    return [dim, pos], vel

    # NOTE(ycho-or): see below for no-collision scene generation.
    # objects = []
    # bboxes = []
    # for i in range(num_objects):
    #     for _ in range(config.max_gen_iterations):
    #         # Generate random vehicle.
    #         vehicle, velocity = create_vehicle((), config)
    #         dim, pose = vehicle
    #         bbox = get_bounding_box(dim, pose)

    #         # Check collision with existing objects.
    #         for prev_bbox in bboxes:
    #             if bbox_intersects(prev_bbox, bbox):
    #                 continue

    #         # Append object to result.
    #         objects.append((vehicle, velocity))
    #         bboxes.append(bbox)
    #         break
    return objects


def random_rotation_matrix(size: tr.Size):
    # Unit vector.
    theta = tr.acos(2.0 * tr.rand(size) - 1.0)
    phi = (2 * np.pi) * tr.rand(size)
    ct, st = tr.cos(theta), tr.sin(theta)
    cp, sp = tr.cos(phi), tr.sin(phi)
    rvec = tr.stack([cp*ct, cp*st, sp], axis=-1)

    # Angle mag.
    angle = (2 * np.pi) * tr.rand(size)
    rvec *= angle[..., None]

    rmat = kornia.angle_axis_to_rotation_matrix(rvec.view(-1, 3))
    return rmat.view(size + (3, 3))


def create_rays(pose: tr.Tensor, velocity: tr.Tensor, stamp: tr.Tensor, config: Config):
    # Currently pose/velocity are both assumed 2D.
    # NOTE(yycho0108): Currently ray transform is coincident to vehicle transform.
    # Consider applying offsets here instead ?
    ray_pose = pose[None, None, :] + \
        velocity[None, None, :] * tr.from_numpy(config.stamps[..., None]).cuda()
    # Convert 2D (x,y) -> 3D (x,y,z) Ray origin
    ray_origin = tr.cat((ray_pose[...,:2], tr.tensor(config.ray_z).view(1,1).float()), -1)
    
    # ray_origin = np.insert(ray_pose[..., :2], 2, config.ray_z, axis=-1)

    # Rotate vector according to pose

    #rvec = np.zeros_like(ray_pose)
    #rvec[..., :2] = 0
    #r = R.from_rotvec(rvec.reshape(-1, 3))
    #ray_direction = r.apply(config.rays.reshape(-1, 3))
    rvec = tr.zeros_like(ray_pose)
    rvec[..., 2] = ray_pose[..., 2]
    rmat = kornia.angle_axis_to_rotation_matrix(rvec.view(-1, 3)).view(
        ray_pose.shape[:-1] + (3, 3)).float()
    # rmat = random_rotation_matrix(ray_pose.shape[:2])
    rays_0 = tr.from_numpy(config.rays)
    ray_dirs = tr.einsum('...ab,...b->...a', rmat, rays_0)
    ray_dirs = ray_dirs.view(config.rays.shape)
    # ray_direction = ray_direction.reshape(config.rays.shape)

    # Format output and return.
    rays = (ray_origin, ray_dirs)
    return (rays, tr.from_numpy(config.stamps) + stamp)


def get_axes(h):
    shape = h.shape + (3, 3)
    axes = np.zeros(shape)
    c, s = np.cos(h), np.sin(h)
    axes[..., 0, 0] = c
    axes[..., 0, 1] = -s
    axes[..., 1, 0] = s
    axes[..., 1, 1] = c
    axes[..., 2, 2] = 1
    return axes


def cube_signs():
    return np.asarray([(1, -1, 1),
                       (1, -1, -1),
                       (1, 1, -1),
                       (1, 1, 1),
                       (-1, -1, 1),
                       (-1, -1, -1),
                       (-1, 1, -1),
                       (-1, 1, 1)], dtype=np.int32)


def cube_indices():
    return np.asarray([(4, 0, 3),
                       (4, 3, 7),
                       (0, 1, 2),
                       (0, 2, 3),
                       (1, 5, 6),
                       (1, 6, 2),
                       (5, 4, 7),
                       (5, 7, 6),
                       (7, 3, 2),
                       (7, 2, 6),
                       (0, 5, 1),
                       (0, 4, 5)], dtype=np.int32)


def get_vertices(dim: tr.Tensor, poses: tr.Tensor, config: Config):
    # Cache input shapes.
    ds = dim.shape
    ps = poses.shape

    # Create canonial bounding box from dimensions.
    bbox = 0.5 * dim[:, None] * tr.from_numpy(cube_signs())[None, :]
    # bbox -> O83

    # Extract transform from pose.
    poses = poses.view((-1,) + poses.shape[-2:])  # -> (R, O, 3)
    rvec = tr.tensor([0, 0, 1]).view(1, 1, 3) * poses
    rmat = kornia.angle_axis_to_rotation_matrix(
        rvec.view(-1, 3)).view(poses.shape[:-1] + (3, 3))
    # rmat -> (RO33)

    # Apply transform.
    bbox = tr.einsum('abde,bce->abcd', rmat.float(), bbox)  # RO83
    bbox[..., :2] += poses[:, :, None, :2]
    bbox[..., 2] += 0.5*dim[None, :, None, 2]  # lift bbox up (s.t. zmin=0)

    # Extract triangles from cube.
    indices = tr.from_numpy(cube_indices())
    vertices = bbox[:, :, indices.long()]  # (R,O,12,3,3)
    return vertices


def get_triangles(stamps, scene, config: Config):
    # Extract scene.
    (dim, pos0), vel = scene

    # Apply motion to initial pose, based on velocity.
    pos = pos0.view(1, -1, 3) + vel.view(1, -1, 3) * stamps.view(-1, 1, 1)
    pos = pos.view(stamps.shape + pos0.shape)  # => (R,O,3)
    vertices = get_vertices(dim, pos, config)

    # Format result and return.
    return vertices.reshape(-1, 3, 3)


def raytrace(rays, stamps, scene, config):
    ray_origin, ray_vector = rays
    triangles = get_triangles(stamps, scene, config)
    triangles = triangles.reshape(ray_origin.shape[:-1] + (-1, 3, 3))
    hits, dists = ray_triangle_intersection(
        ray_origin.float(),
        ray_vector.float(),
        triangles.float(), broadcast_triangle=False,
        max_distance=config.max_ray_distance
    )
    return hits, dists


def apply_pose(cloud, stamp, stamps, pose, velocity, config: Config):
    # Extract transforms.
    dt = stamps - stamp
    print(F"ps:{pose.shape}")
    print(F"vs:{velocity.shape}")
    print(F"dt:{dt.shape}")
    pose_at_stamp = pose[None, None, :] + \
        dt[..., None] * velocity[None, None, :]

    # Rotation about z axis
    rvec = tr.zeros_like(pose_at_stamp)
    rvec[..., 2] = pose_at_stamp[..., 2]
    rmat = kornia.angle_axis_to_rotation_matrix(
        rvec.view(-1, 3)).view(rvec.shape[:-1]+(3, 3)).float()

    # Apply transforms.
    print(rmat.shape, cloud.shape)
    cloud = tr.einsum('...ab,...b->...a', rmat, cloud)
    cloud[..., :2] += pose[..., :2]
    cloud[..., 2] += config.ray_z
    return cloud


def main():
    tr.set_default_tensor_type('torch.cuda.FloatTensor')

    config = create_config()
    scene = create_scene(config)
    (dim, pose), velocity = create_vehicle((), config)

    # Zero out pose for debugging purposes
    # pose *= 0.0

    data_queue, event_queue, command_queue = ProxyViewer.create()
    command_queue.put(AddGridCommand(
        name='grid', size=(400, 400, 1), spacing=(10, 10, 10)))
    command_queue.put(AddPointsCommand(name='cloud'))
    command_queue.put(AddLinesCommand(name='obstacles'))
    command_queue.put(AddPointsCommand(name='self'))

    while True:
        rays, stamps = create_rays(pose, velocity, 0.0, config)
        hits, dists = raytrace(rays, stamps, scene, config)
        range_image = tr.where(hits, dists, tr.full_like(dists, float('inf')))
        range_image = tr.min(range_image, dim=-1).values
        rim = range_image.numpy()

        cloud = tr.from_numpy(config.rays) * range_image[..., None]
        cloud = apply_pose(cloud, 0.0, stamps, pose, velocity, config)
        cloud = cloud.numpy().reshape(-1, 3)

        obstacle_vertices = get_vertices(*scene[0], config).reshape(-1, 3, 3)
        obstacle_vertices = obstacle_vertices.numpy()
        lines = obstacle_vertices[:, [
            (0, 1), (1, 2), (2, 0)], :].reshape(-1, 2, 3)

        vpos = pose[..., :2].numpy()
        vpos = np.asarray([[vpos[0], vpos[1], config.ray_z]])

        data_queue.put(dict(
            cloud=dict(pos=cloud.reshape(-1, 3)),
            obstacles=dict(pos=lines),
            self=dict(pos=vpos, color=(1, 0, 0, 1), size=10.0)
        ))

        # Apply velocity
        pose += velocity * config.win
        scene[0][1] += scene[1] * config.win

        time.sleep(0.001)

    #ax = plt.gca(projection='3d')
    #ax.plot([vpos[0]], [vpos[1]], [0], 'ko')
    #ax.plot(cloud[..., 0], cloud[..., 1], cloud[..., 2], 'r+')

    ##corners = get_bounding_box(*scene[0]).numpy()
    # for c in corners:
    ##    ax.plot(c[..., 0], c[..., 1], 0*c[..., 1] - config.ray_z, 'b--')

    # obstacle_vertices = get_vertices(*scene[0], config).reshape(-1, 3, 3)
    #obstacle_vertices = obstacle_vertices.numpy()
    #lines = obstacle_vertices[:, [(0, 1), (1, 2), (2, 0)], :].reshape(-1, 2, 3)
    # print(lines.shape)
    #lc = Line3DCollection(lines)

    # ax.add_collection(lc)

    print(F"${obstacle_vertices.shape}")

    # plt.axis('equal')
    # plt.grid()

    # plt.imshow(1.0/rim)
    # plt.show()


if __name__ == '__main__':
    main()
