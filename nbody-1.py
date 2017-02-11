"""
    N-body simulation.

    Name: Danny Vilela
    NetID: dov205

    Original runtime: Average(76s, 76, 75s) = 75.666s
    Improved runtime: Average(30.6s, 31.5s, 30.6) = 30.9s
    Relative speedup: 75.666 / 30.9 = 2.448x
"""

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
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


# DEPRECATED WITH CHANGE (2)
def compute_deltas(x1, x2, y1, y2, z1, z2):
    return (x1-x2, y1-y2, z1-z2)


# DEPRECATED WITH CHANGE (1)
def compute_b(m, dt, dx, dy, dz):
    mag = compute_mag(dt, dx, dy, dz)
    return m * mag


# DEPRECATED WITH CHANGE (1)
def compute_mag(dt, dx, dy, dz):
    return dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))


# DEPRECATED WITH CHANGE(1)
def update_vs(v1, v2, dt, dx, dy, dz, m1, m2):
    v1[0] -= dx * compute_b(m2, dt, dx, dy, dz)
    v1[1] -= dy * compute_b(m2, dt, dx, dy, dz)
    v1[2] -= dz * compute_b(m2, dt, dx, dy, dz)
    v2[0] += dx * compute_b(m1, dt, dx, dy, dz)
    v2[1] += dy * compute_b(m1, dt, dx, dy, dz)
    v2[2] += dz * compute_b(m1, dt, dx, dy, dz)


# DEPRECATED WITH CHANGE (5)
def update_rs(r, dt, vx, vy, vz):
    r[0] += dt * vx
    r[1] += dt * vy
    r[2] += dt * vz


def advance(dt, iterations):
    '''
        advance the system one timestep
    '''

    # CHANGE (4): Remove :iterations calls to advance, and
    # just do it :iterations times in advance()
    for _ in range(iterations):
        seenit = []
        for body1 in BODIES.keys():
            for body2 in BODIES.keys():
                if (body1 != body2) and not (body2 in seenit):
                    ([x1, y1, z1], v1, m1) = BODIES[body1]
                    ([x2, y2, z2], v2, m2) = BODIES[body2]

                    # CHANGE (2): Omit compute_*, instead just perform subtractions
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

                    seenit.append(body1)
        
        for body in BODIES.keys():
            (r, [vx, vy, vz], m) = BODIES[body]

            # CHANGE (5): Remove call to update_rs and just do it
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz


# DEPRECATED WITH CHANGE (2)
def compute_energy(m1, m2, dx, dy, dz):
    return (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)


def report_energy(e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''

    seenit = []
    for body1 in BODIES.keys():
        for body2 in BODIES.keys():
            if (body1 != body2) and not (body2 in seenit):
                ((x1, y1, z1), v1, m1) = BODIES[body1]
                ((x2, y2, z2), v2, m2) = BODIES[body2]

                # CHANGE (2): Omit compute_*, instead just perform computations
                (dx, dy, dz) = (x1 - x2, y1 - y2, z1 - z2)
                e -= ((m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5))

                seenit.append(body1)
        
    for body in BODIES.keys():
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

def offset_momentum(ref, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''

    for body in BODIES.keys():
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''

    # Set up global state
    # CHANGE (6): Remove one-off call to offset_momentum and nest it within nbody.
    px, py, pz = 0, 0, 0

    for body in BODIES.keys():
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m

    (r, v, m) = BODIES[reference]
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m

    for _ in range(loops):
        report_energy()

        # CHANGE (4): Remove :iterations calls to advance, and
        # just do it once.
        advance(0.01, iterations)

        print(report_energy())

if __name__ == '__main__':
    nbody(100, 'sun', 20000)

