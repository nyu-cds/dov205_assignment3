"""
    N-body simulation.

    Name: Danny Vilela
    NetID: dov205

    Original runtime: Average(76s, 76, 75s) = 75.666s
    Improved runtime: Average(25.4s, 25.4s, 25.4s) = 25.4s -- under 30s!
    Relative speedup: (75.666 / 25.4) = 2.978x
"""


# CHANGE (3.1): Added BODIES as local variable.
def advance(dt, iterations, BODIES, BODY_NAMES):
    '''
        advance the system one timestep
    '''

    # CHANGE (1.4): Remove :iterations calls to advance, and
    # just do it :iterations times in advance()
    for _ in range(iterations):

        # CHANGE (2.1): Make seenit a set, for O(1) lookup/membership
        seenit = set()

        # CHANGE (4.2): Change collected from BODIES.keys() to :body_names
        for body1 in BODY_NAMES:
            for body2 in BODY_NAMES:

                if (body1 != body2) and not (body2 in seenit):

                    ([x1, y1, z1], v1, m1) = BODIES[body1]
                    ([x2, y2, z2], v2, m2) = BODIES[body2]

                    # CHANGE (1.2): Omit compute_*, instead just perform subtractions
                    (dx, dy, dz) = (x1 - x2, y1 - y2, z1 - z2)

                    # CHANGE (1): Remove function calls to compute_b and look-aside calls to
                    #  compute_mag. We can also store values that are computed many times,
                    #  like inner_mag, mag_m2, and mag_m1.
                    inner_mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
                    b_m2 = m2 * inner_mag
                    b_m1 = m1 * inner_mag
                    v1[0] -= dx * b_m2
                    v1[1] -= dy * b_m2
                    v1[2] -= dz * b_m2
                    v2[0] += dx * b_m1
                    v2[1] += dy * b_m1
                    v2[2] += dz * b_m1

                    # CHANGE (2.1): Update .append() to .add()
                    seenit.add(body1)

        # CHANGE (4.2): Change collected from BODIES.keys() to :body_names
        for body in BODY_NAMES:
            (r, [vx, vy, vz], m) = BODIES[body]

            # CHANGE (1.5): Remove call to update_rs and just do it
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz


# CHANGE (3.1): Added BODIES as local variable.
def report_energy(BODIES, BODY_NAMES, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''

    # CHANGE (2.1): Make seenit a set, for O(1) lookup/membership
    seenit = set()

    # CHANGE (4.2): Change collected from BODIES.keys() to :body_names
    for body1 in BODY_NAMES:
        for body2 in BODY_NAMES:

            if (body1 != body2) and not (body2 in seenit):
                ((x1, y1, z1), v1, m1) = BODIES[body1]
                ((x2, y2, z2), v2, m2) = BODIES[body2]

                # CHANGE (1.2): Omit compute_*, instead just perform computations
                (dx, dy, dz) = (x1 - x2, y1 - y2, z1 - z2)
                e -= ((m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5))

                # CHANGE (1): Update .append() to .add()
                seenit.add(body1)

    # CHANGE (4.2): Change collected from BODIES.keys() to :body_names
    for body in BODY_NAMES:
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e


# CHANGE (3.1): Added BODIES as local variable.
def nbody(loops, reference, iterations, BODIES):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''

    # Set up global state
    # CHANGE (1.6): Remove one-off call to offset_momentum and nest it within nbody.
    px, py, pz = 0, 0, 0

    # CHANGE (1.6): Also introduce body_names at nbody so we can weave it throughout state.
    BODY_NAMES = BODIES.keys()

    for body in BODY_NAMES:
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m

    (r, v, m) = BODIES[reference]
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m

    for _ in range(loops):

        report_energy(BODIES, BODY_NAMES)

        # CHANGE (1.4): Remove :iterations calls to advance, and
        # just do it once.
        advance(0.01, iterations, BODIES, BODY_NAMES)

        print(report_energy(BODIES, BODY_NAMES))

if __name__ == '__main__':

    # CHANGE (3.2): Added global variables to local w.r.t main()
    SOLAR_MASS = 4 * 3.14159265358979323 * 3.14159265358979323
    DAYS_PER_YEAR = 365.24
    BODIES = {
        'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),

        'jupiter': ([4.84143144246472090e+00,
                     -1.16032004402742839e+00,
                     -1.03622044471123109e-01],
                    [1.66007664274403694e-03 * DAYS_PER_YEAR,
                     7.69901118419740425e-03 * DAYS_PER_YEAR,
                     -6.90460016972063023e-05 * DAYS_PER_YEAR],
                    9.54791938424326609e-04 * SOLAR_MASS),

        'saturn': ([8.34336671824457987e+00,
                    4.12479856412430479e+00,
                    -4.03523417114321381e-01],
                   [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                    4.99852801234917238e-03 * DAYS_PER_YEAR,
                    2.30417297573763929e-05 * DAYS_PER_YEAR],
                   2.85885980666130812e-04 * SOLAR_MASS),

        'uranus': ([1.28943695621391310e+01,
                    -1.51111514016986312e+01,
                    -2.23307578892655734e-01],
                   [2.96460137564761618e-03 * DAYS_PER_YEAR,
                    2.37847173959480950e-03 * DAYS_PER_YEAR,
                    -2.96589568540237556e-05 * DAYS_PER_YEAR],
                   4.36624404335156298e-05 * SOLAR_MASS),

        'neptune': ([1.53796971148509165e+01,
                     -2.59193146099879641e+01,
                     1.79258772950371181e-01],
                    [2.68067772490389322e-03 * DAYS_PER_YEAR,
                     1.62824170038242295e-03 * DAYS_PER_YEAR,
                     -9.51592254519715870e-05 * DAYS_PER_YEAR],
                    5.15138902046611451e-05 * SOLAR_MASS)}

    # CHANGE (3.1): Weave BODIES through the program state
    nbody(100, 'sun', 20000, BODIES)

