import numpy as np

def jitter_pc(p_random_state, p_pts, p_noise=0.001, p_clip=0.005):
    """Method to jitter a point cloud.

    Args:
        p_random_state (np.RandomState): Random state.
        p_pts (float np.array nxd): Point cloud.
        p_noise (float): Noise added to the point coordinates.
        p_clip (float): Value to clip the noise.
    Returns:
        (float np.array nxd): Mirrored points.
    """
    if p_clip > 0.0:
        noise = np.clip(p_random_state.randn(p_pts.shape[0], p_pts.shape[1])*p_noise,
            -1.0*p_clip, p_clip)
    else:
        noise = p_random_state.randn(p_pts.shape[0], p_pts.shape[1])*p_noise

    return p_pts + noise


def mirror_pc(p_random_state, p_axis, p_pts, p_normals=None, p_return_mirror = False):
    """Method to mirror axis of the point cloud.

    Args:
        p_random_state (np.RandomState): Random state.
        p_axis (bool list): Boolean list that indicates which axis can 
            be mirrored.
        p_pts (float np.array nxd): Point cloud.
        p_normals (float n.parray nxd): Point normals.
    Returns:
        (float np.array nxd): Mirrored points.
        (float np.array nxd): Mirrored normals.
    """
    mul_vals = np.full((1, p_pts.shape[1]), 1.0, dtype=np.float32)
    for cur_axis  in range(len(p_axis)):
        if p_random_state.random_sample() > 0.5 and p_axis[cur_axis]:
            mul_vals[0, cur_axis] = -1.0
    ret_pts = p_pts*mul_vals
    ret_normals = None
    if not(p_normals is None):
        ret_normals = p_normals*mul_vals
    if p_return_mirror:
        return ret_pts, ret_normals, mul_vals
    else:
        return ret_pts, ret_normals


def anisotropic_scale_pc(p_random_state, p_pts, 
    p_min_scale = 0.9, p_max_scale = 1.1, p_return_scaling = False):
    """Method to scale a model with anisotropic scaling.

    Args:
        p_random_state (np.RandomState): Random state.
        p_pts (float np.array nxd): Point cloud.
        p_min_scale (float): Minimum scaling.
        p_max_scale (float): Maximum scaling.
        p_return_scaling (bool): Boolean that indicates if 
            the scaling is also returned.
    Returns:
        (float tensor nxd): Scaled point cloud.
    """

    #Total scaling range.
    scale_range = p_max_scale - p_min_scale
    cur_scaling = p_random_state.random_sample(
        (1, p_pts.shape[1]))*scale_range + p_min_scale

    #Return the scaled points.
    if p_return_scaling:
        return p_pts*cur_scaling, cur_scaling
    else:
        return p_pts*cur_scaling


def rotate_pc_3d(p_random_state, p_pts, p_max_angle = 2.0 * np.pi, 
    p_axes = [0, 1, 2]):
    """Method to rotate a 3D point cloud.

    Args:
        p_random_state (np.RandomState): Random state.
        p_pts (float np.array nx3): Point cloud.
        p_max_angle (float): Max rotation angle.
        p_axes (list of ints): Axis for which we compute the rotation 
            (0:X, 1:Y, 2:Z).
    Returns:
        (float np.array nx3): Rotated point cloud.
        (float np.array nx3): Rotated point cloud normals.
        (float np.array 3x3): Rotation matrix.
    """

    #Compute the rotation matrix.
    angles = (p_random_state.random_sample((3))-0.5) * 2.0 * p_max_angle
    if 0 in p_axes:
        Rx = np.array([[1.0, 0.0, 0.0],
                [0.0, np.cos(angles[0]), -np.sin(angles[0])],
                [0.0, np.sin(angles[0]), np.cos(angles[0])]])
    else:
        Rx = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    if 1 in p_axes:
        Ry = np.array([[np.cos(angles[1]), 0.0, np.sin(angles[1])],
                [0.0, 1.0, 0.0],
                [-np.sin(angles[1]), 0.0, np.cos(angles[1])]])
    else:
        Ry = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    if 2 in p_axes:
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0.0],
                [np.sin(angles[2]), np.cos(angles[2]), 0.0],
                [0.0, 0.0, 1.0]])
    else:
        Rz = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    rotation_matrix = np.dot(Rz, np.dot(Ry,Rx))

    #Compute the rotated point cloud.
    return np.dot(p_pts, rotation_matrix), rotation_matrix


def uniform_samples_surface_sphere(p_random_state, p_num_pts, p_num_dims):
    """Method to uniform sample the surface of a sphere.

    Args:
        p_random_state (np.RandomState): Random state.
        p_num_pts (int): Number of points.
        p_num_dims (int): Number of dimensions of each point.
    Returns:
        (float np.array nxd): New point cloud.
    """

    rnd_pts = p_random_state.normal(0.0, 1.0, [p_num_pts, p_num_dims])
    rnd_pts = rnd_pts/ np.linalg.norm(rnd_pts, axis=1)[:, np.newaxis]
    return rnd_pts.astype(np.float32)


def uniform_samples_sphere(p_random_state, p_num_pts, p_num_dims):
    """Method to uniform sample the volume of a sphere.

    Args:
        p_random_state (np.RandomState): Random state.
        p_num_pts (int): Number of points.
        p_num_dims (int): Number of dimensions of each point.
    Returns:
        (float np.array nxd): New point cloud.
    """

    rnd_pts = uniform_samples_surface_sphere(p_random_state, p_num_pts, p_num_dims)
    u_var = np.power(p_random_state.uniform(0.0, 1.0, [p_num_pts, 1]), 1.0/float(p_num_dims))
    return (rnd_pts * u_var).astype(np.float32)


def poisson_disk_sampling(p_pts, p_radii):
    """Method to sample a set of point using poisson disk sampling.

    Args:
        p_pts (float np.array nxd): Input points.
        p_radii (float np.array d): List of radii in each dimension. 
    Returns:
        (float np.array nx3): New point cloud.
    """

    diffs = (p_pts[:, np.newaxis, :] - p_pts[np.newaxis, :, :])/p_radii
    dists = np.linalg.norm(diffs, axis=2)

    selected_pts = [0]
    for cur_pt in range(1, len(p_pts)):
        if np.amin(dists[cur_pt, selected_pts]) >= 1.0:
            selected_pts.append(cur_pt)

    return p_pts[selected_pts, :]


def farthest_point_sampling(p_pts, p_num_pts, p_return_indices = False):
    """Method to sample a set of points using farthest point sampling.

    Args:
        p_pts (float np.array nxd): Input points.
        p_num_pts (int): Number of desired points. 
        p_return_indices (bool): Boolean that indicates if the indices are returned.
    Returns:
        (float np.array nx3): New point cloud.
    """

    indices = [0]#[int(p_random_state.rand() *(len(p_pts)-1))]
    cur_point = p_pts[indices[0]]
    dist = np.linalg.norm(p_pts - cur_point, axis = 1)
    while len(indices) < p_num_pts:
        curr_ind = np.argmax(dist)
        indices.append(curr_ind)
        cur_point = p_pts[curr_ind]
        dist = np.amin(np.concatenate((dist.reshape(-1, 1), \
            np.linalg.norm(p_pts-cur_point, axis = 1).reshape(-1, 1)), axis=1), axis=1)
            
    if p_return_indices:
        return p_pts[indices, :], indices
    else:
        return p_pts[indices, :]