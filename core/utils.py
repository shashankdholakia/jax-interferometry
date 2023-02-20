import jax.numpy as jnp
from jaxopt import Bisection
import matplotlib.pyplot as plt
from jax.lax import cond
import jax
from jax import grad, jit, vmap


@jit
def closest_polygon(thetas):
    n = jnp.rint(jnp.pi / jnp.arcsin(jnp.pi/(2*(len(thetas))*jnp.sin(thetas)))).astype(int)
    n = jnp.where(n==0,
                 1,
                 n)
    return n

#cannot be jitted due to array shapes that depend on inputs
#alternative may be to instatiate a jitted version for a given grid size?
def revolve(thetas, rtws, Ts, Fs, ns):
    rtw = rtws.repeat(ns)
    T = Ts.repeat(ns)
    F = Fs.repeat(ns)
    theta = thetas.repeat(ns)
    return jnp.array([rtw,T, F, theta])

@jit
def spherical_to_cartesian(r, theta, phi):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.cos(theta)
    z = r * jnp.sin(theta) * jnp.sin(phi)
    return jnp.array([x, y, z])

@jit
def rotate_point_cloud(points, inclination, obliquity):
    # define the rotation matrices
    R_x = jnp.array([[1, 0, 0],
                     [0, jnp.cos(inclination), -jnp.sin(inclination)],
                     [0, jnp.sin(inclination), jnp.cos(inclination)]])
    R_z = jnp.array([[jnp.cos(obliquity), -jnp.sin(obliquity), 0],
                     [jnp.sin(obliquity), jnp.cos(obliquity), 0],
                     [0, 0, 1]])
    # rotate the point cloud
    points_rotated = jnp.dot(points, R_x)
    points_rotated = jnp.dot(points_rotated, R_z)
    return points_rotated

@jit
def triangle_normals(points, triangulation):
    a = points[triangulation[:,0],:]
    b = points[triangulation[:,1],:]
    c = points[triangulation[:,2],:]
    # compute the normal vectors
    normals = jnp.cross(b-a, c-a)
    # compute the center of the triangle
    center = (a + b + c) / 3
    # reverse the normal vector if the dot product is negative
    normals = normals*jnp.sign(jnp.sum(normals*center, axis=1))[:,jnp.newaxis]
    return normals

@jit
def barycenter(points, triangulation):
    # get the coordinates of the triangle vertices
    x = points[triangulation, 0]
    y = points[triangulation, 1]
    z = points[triangulation, 2]
    # compute the barycenter coordinates
    x_barycenter = jnp.mean(x, axis=1)
    y_barycenter = jnp.mean(y, axis=1)
    z_barycenter = jnp.mean(z, axis=1)
    # stack the barycenter coordinates into an array
    barycenters = jnp.stack((x_barycenter, y_barycenter, z_barycenter), axis=1)
    return barycenters

@jit
def triangle_area(points, triangulation):
    a = points[triangulation[:,0],:]
    b = points[triangulation[:,1],:]
    c = points[triangulation[:,2],:]
    # compute the edges of the triangle
    u = b - a
    v = c - a
    # compute the cross product
    w = jnp.cross(u, v)
    # compute the area using the formula
    area = 0.5 * jnp.linalg.norm(w, axis=1)
    return area
