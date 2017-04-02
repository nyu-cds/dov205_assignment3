from itertools import combinations
from numba import jit, int32, float64, vectorize
import numpy as np

"""
    N-body simulation.

    Name: Danny Vilela
    NetID: dov205

    Original runtime: Average(76s, 76, 75s) = 75.666s
    Improved runtime: Average(22.6s, 22.4s, 22.2s) = 22.4s -- under 30s!
    Relative speedup: (75.666 / 22.4) = 3.377x
"""

@vectorize([float64(float64, float64)])
def vec_deltas(a, b):
    """Takes two floating point vectors and performs pairwise subtraction."""
    return a - b


@jit('void(float64, int32, float64[:,:,:], char[:], char[:,:])')
def advance(dt, iterations, bodies, body_names, key_pairs):
    """Advance the system :dt time, :iterations times.

    :param dt: the change in time between the previous time and now
    :param iterations: the number of times to do :dt advances in one simulation.
    :param bodies: {name : body_information} dictionary for all bodies
    :param body_names: names of bodies (= bodies.keys())
    :param key_pairs: pairs of bodies in :bodies' keys
    """

    for _ in range(iterations):

        for (body1, body2) in key_pairs:

            (p1, v1, m1) = bodies[body1]
            (p2, v2, m2) = bodies[body2]

            dx, dy, dz = vec_deltas(p1, p2)

            inner_mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            b_m2 = m2 * inner_mag
            b_m1 = m1 * inner_mag
            
            v1[0] -= dx * b_m2
            v1[1] -= dy * b_m2
            v1[2] -= dz * b_m2
            v2[0] += dx * b_m1
            v2[1] += dy * b_m1
            v2[2] += dz * b_m1

        for body in body_names:

            (r, [vx, vy, vz], m) = bodies[body]

            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz


@jit('void(float64[:,:,:], char[:], char[:,:], float64)')
def report_energy(bodies, body_names, key_pairs, e=0.0):
    """Compute the energy and return it so that it can be printed

    :param bodies: {name : body_information} dictionary for all bodies
    :param body_names: names of bodies (= bodies.keys())
    :param key_pairs: pairs of bodies in :bodies' keys
    :param e: baseline energy
    :return: e
    """

    for (body1, body2) in key_pairs:
        (p1, v1, m1) = bodies[body1]
        (p2, v2, m2) = bodies[body2]

        (dx, dy, dz) = vec_deltas(p1, p2)
        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
        
    for body in body_names:
        (r, [vx, vy, vz], m) = bodies[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e


@jit('void(int32, char, int32, float64[:,:,:])')
def nbody(loops, reference, iterations, bodies):
    """N-body simulation.

    :param loops: number of loops to run
    :param reference: body at center of system
    :param iterations: number of timesteps to advance
    :param bodies: {name : body_information} dictionary for all bodies
    """

    # Set up global state
    px, py, pz = 0, 0, 0

    body_names = bodies.keys()
    
    # Chose to deprecate product due to Numba's inability to compile correctly.
    # We previously got the error:
    #     DataFlowAnalysis' object has no attribute 'op_MAKE_FUNCTION'
    body_pairs = list(combinations(body_names, 2))

    for body in body_names:
        (r, [vx, vy, vz], m) = bodies[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m

    (r, v, m) = bodies[reference]
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m

    for _ in range(loops):

        advance(0.01, iterations, bodies, body_names, body_pairs)

        print(report_energy(bodies, body_names, body_pairs))


if __name__ == '__main__':

    PI = 3.14159265358979323
    SOLAR_MASS = 4 * PI * PI
    DAYS_PER_YEAR = 365.24

    bodies = {
        'sun': (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), SOLAR_MASS),

        'jupiter': (np.array([4.84143144246472090e+00,
                     -1.16032004402742839e+00,
                     -1.03622044471123109e-01]),
                    np.array([1.66007664274403694e-03 * DAYS_PER_YEAR,
                     7.69901118419740425e-03 * DAYS_PER_YEAR,
                     -6.90460016972063023e-05 * DAYS_PER_YEAR]),
                    9.54791938424326609e-04 * SOLAR_MASS),

        'saturn': (np.array([8.34336671824457987e+00,
                    4.12479856412430479e+00,
                    -4.03523417114321381e-01]),
                   np.array([-2.76742510726862411e-03 * DAYS_PER_YEAR,
                    4.99852801234917238e-03 * DAYS_PER_YEAR,
                    2.30417297573763929e-05 * DAYS_PER_YEAR]),
                   2.85885980666130812e-04 * SOLAR_MASS),

        'uranus': (np.array([1.28943695621391310e+01,
                    -1.51111514016986312e+01,
                    -2.23307578892655734e-01]),
                   np.array([2.96460137564761618e-03 * DAYS_PER_YEAR,
                    2.37847173959480950e-03 * DAYS_PER_YEAR,
                    -2.96589568540237556e-05 * DAYS_PER_YEAR]),
                   4.36624404335156298e-05 * SOLAR_MASS),

        'neptune': (np.array([1.53796971148509165e+01,
                     -2.59193146099879641e+01,
                     1.79258772950371181e-01]),
                    np.array([2.68067772490389322e-03 * DAYS_PER_YEAR,
                     1.62824170038242295e-03 * DAYS_PER_YEAR,
                     -9.51592254519715870e-05 * DAYS_PER_YEAR]),
                    5.15138902046611451e-05 * SOLAR_MASS)
    }

    nbody(100, 'sun', 20000, bodies)