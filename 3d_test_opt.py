
import numpy as np

#############################
# OWN function definition START


def get_random_distributed_series(start, end, n_samples, fix_seed=True):
    if fix_seed:
        np.random.seed(0)
    # generate randomly spaced data
    x = np.random.random(n_samples).cumsum()
    # rescale to desired range
    x = (x - x.min()) / x.ptp()
    x = (end - start) * x + start
    return x


def map_forces_to_nodes(nodal_coordinates, nodes_geom_center, target_resultants):

    n_nodes = len(nodal_coordinates)

    mapping_coef_matrix = np.zeros((6, 3*n_nodes))

    # fx contribution
    mapping_coef_matrix[0, 0::3] = 1.0
    # fy contribution
    mapping_coef_matrix[1, 1::3] = 1.0
    # fz contribution
    mapping_coef_matrix[2, 2::3] = 1.0
    # mx, my, mz
    for i in range(0, n_nodes):
        # dx, dy, dz
        dx = nodal_coordinates[i][0]-nodes_geom_center[0]
        dy = nodal_coordinates[i][1]-nodes_geom_center[1]
        dz = nodal_coordinates[i][2]-nodes_geom_center[2]

        # mx = dy * fz - dz * fy
        # fy contribution
        mapping_coef_matrix[3, 3*i+1] = -dz
        # fz contribution
        mapping_coef_matrix[3, 3*i+2] = dy

        # my = dz * fx - dx * fz
        # fx contribution
        mapping_coef_matrix[4, 3*i+0] = dz
        # fz contribution
        mapping_coef_matrix[4, 3*i+2] = -dx

        # mz = dx * fy - dy * fx
        # fx contribution
        mapping_coef_matrix[5, 3*i+0] = -dy
        # fy contribution
        mapping_coef_matrix[5, 3*i+1] = dx

    # find maximum distance
    # to be used as a norming factor
    # for distance scaling
    x_max = 0.0
    y_max = 0.0
    z_max = 0.0

    for i in range(0, n_nodes):
        x_max_new = abs(nodal_coordinates[i][0]-nodes_geom_center[0])
        if x_max_new > x_max:
            x_max = x_max_new

        y_max_new = abs(nodal_coordinates[i][1]-nodes_geom_center[1])
        if y_max_new > y_max:
            y_max = y_max_new

        z_max_new = abs(nodal_coordinates[i][2]-nodes_geom_center[2])
        if z_max_new > z_max:
            z_max = z_max_new


    def evaluate_current_resultant_and_nodal_forces(mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs):
        '''
        A function to evaluate the current resultant
        based on assumed nodal forces
        
            Fx, Fy, Fz -> split uniformly between the number of nodes
        
            Mx, My, Mz -> split uniformly between the number of nodes
            also taking care of a linear relation as a function
            of the distance from the center
        '''
        
        current_nodal_forces = np.zeros(3*n_nodes)
        for i in range(0, n_nodes):
            # fx
            # to Fx
            current_nodal_forces[i*3+0] = coefs[0] * \
                target_resultants[0] / n_nodes
            # to Mz
            current_nodal_forces[i*3+0] += coefs[3] * target_resultants[5] / \
                2 / n_nodes * \
                (nodal_coordinates[i][1]-nodes_geom_center[1])/y_max
            # to My
            current_nodal_forces[i*3+0] += coefs[4] * target_resultants[4] / \
                2 / n_nodes * \
                (nodal_coordinates[i][2]-nodes_geom_center[2])/z_max

            # fy
            # to Fy
            current_nodal_forces[i*3+1] = coefs[1] * \
                target_resultants[1] / n_nodes
            # to Mz
            current_nodal_forces[i*3+1] += coefs[5] * target_resultants[5] / \
                2 / n_nodes * \
                (nodal_coordinates[i][0]-nodes_geom_center[0])/x_max
            # to Mx
            current_nodal_forces[i*3+1] += coefs[6] * target_resultants[3] / \
                2 / n_nodes * \
                (nodal_coordinates[i][2]-nodes_geom_center[2])/z_max

            # fz
            # to Fz
            current_nodal_forces[i*3+2] = coefs[2] * \
                target_resultants[2] / n_nodes
            # to Mx
            current_nodal_forces[i*3+2] += coefs[7] * target_resultants[3] / \
                2 / n_nodes * \
                (nodal_coordinates[i][1]-nodes_geom_center[1])/y_max
            # to My
            current_nodal_forces[i*3+2] += coefs[8] * target_resultants[4] / \
                2 / n_nodes * \
                (nodal_coordinates[i][0]-nodes_geom_center[0])/x_max

        current_resultant = np.dot(mapping_coef_matrix, current_nodal_forces)

        return current_resultant, current_nodal_forces

    def get_norm_of_residual(mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs):
        '''
        Return the norm of the current residual
        
        Here the components for Fx, Fy, Mz are weighted higher
        The ones related to Fz, Mx, My are weighted lower
        '''

        res, _ = evaluate_current_resultant_and_nodal_forces(
            mapping_coef_matrix, n_nodes, nodal_coordinates, nodes_geom_center, target_resultants, coefs)

        diff = np.subtract(res, target_resultants)
        diff[0] *= 10
        diff[1] *= 10
        diff[5] *= 10

        diff[2] /= 5
        diff[3] /= 5
        diff[4] /= 5

        return np.linalg.norm(diff)

    from scipy.optimize import minimize
    from functools import partial

    # setting up the optimization problem
    # initially use 1.0 as coefficients to all distributions
    init_coef = [1.0] * 3**2
    initial_norm_of_residual = get_norm_of_residual(mapping_coef_matrix,
                                                        n_nodes,
                                                        nodal_coordinates,
                                                        nodes_geom_center,
                                                        target_resultants,
                                                        init_coef)
    print('Starting optimization')
    print("\tInitial coefficients: " +
          ', '.join([str(val) for val in init_coef]))
    print("\tNorm of residual at this state: " + str(initial_norm_of_residual))
    print()

    # using partial to fix some parameters for the
    optimizable_function = partial(get_norm_of_residual,
                                   mapping_coef_matrix,
                                   n_nodes,
                                   nodal_coordinates,
                                   nodes_geom_center,
                                   target_resultants)

    minimization_result = minimize(optimizable_function,
                                   init_coef,
                                   method='nelder-mead')

    final_coef = minimization_result.x

    final_norm_of_residual = get_norm_of_residual(mapping_coef_matrix,
                                                      n_nodes,
                                                      nodal_coordinates,
                                                      nodes_geom_center,
                                                      target_resultants,
                                                      final_coef)

    print("\tFinal coefficients: " +
          ', '.join([str(val) for val in final_coef]))
    print("\tNorm of residual at this state: " + str(final_norm_of_residual))
    print('Ending optimization')
    print()

    _, nodal_forces = evaluate_current_resultant_and_nodal_forces(mapping_coef_matrix,
                                                                  n_nodes,
                                                                  nodal_coordinates,
                                                                  nodes_geom_center,
                                                                  target_resultants,
                                                                  final_coef)

    return nodal_forces

# OWN function definition END
#############################


#############################
# INPUT of setup START

# dimensions
lx = 32.9
ly = 64.2
lz = 73.5

# offset of starting corner
dx0 = -18.2
dy0 = -72.0
dz0 = 23.3

# sampling points inside the rectangle
nx = 7
ny = 9
nz = 11
n_total = 250

input_forces_geom_center = [lx/2., ly/2., lz/2.]

# predefined setup cases
# NOTE: generic 3D does not work, special cases do work
# NOTE: choose only one of the following 4 as TRUE, preparing cases
# generic 3D -> DOES NOT WORK
case_1 = False
# various special 3D cases mimicking 2D behaviour -> WORK
case_2a = False
case_2b = False
case_2c = False
# generic case with some 0.0 force and moment entries
case_3 = False
# moments generated consistently from forces
case_4 = True

if [case_1, case_2a, case_2b, case_2c, case_3, case_4].count(True) != 1:
    raise Exception('Only 1 case can be and has to be True! Now "True" are: ' +
                    str([case_1, case_2a, case_2b, case_2c, case_3, case_4].count(True)))

# define the input forces, moments and their point of application
if case_1:
    # CASE 1: generic
    input_forces_and_moments = [451.3, 321.2, -512.7, -1545.2, -2117.8, 3001.3]
elif case_2a:
    # CASE 2a: testing 2d-similar case
    input_forces_and_moments = [451.3, 321.2, 0.0, 0.0, 0.0, 3001.3]
elif case_2b:
    # CASE 2b: testing 2d-similar case
    input_forces_and_moments = [451.3, 0.0, -512.7, 0.0, -2117.8, 0.0]
elif case_2c:
    # CASE 2c: testing 2d-similar case
    input_forces_and_moments = [0.0, 321.2, -512.7, -1545.2, 0.0, 0.0]
elif case_3:
    # CASE 3: generic with some 0.0 force and moment entries
    input_forces_and_moments = [451.3, 321.2, 0.0, 0.0, 0.0, 3001.3]
elif case_4:
    # CASE 4: consistent moments with forces
    # create a realistic moment based on force and lever arm
    applied_force = [451.3, 321.2, -512.7]
    applied_at_location = [-78.12, 8.33, 1.52]

    input_forces_and_moments = []
    # Fx, Fy, Fz
    input_forces_and_moments.extend(applied_force)
    # Mx = Dy * Fz - Dz * Fy
    input_forces_and_moments.append((applied_at_location[1]-input_forces_geom_center[1])*applied_force[2] - (
        applied_at_location[2]-input_forces_geom_center[2])*applied_force[1])
    # My = Dz * Fx - Dx * Fz
    input_forces_and_moments.append((applied_at_location[2]-input_forces_geom_center[2])*applied_force[0] - (
        applied_at_location[0]-input_forces_geom_center[0])*applied_force[2])
    # Mz = Dx * Fy - Dy * Fx
    input_forces_and_moments.append((applied_at_location[0]-input_forces_geom_center[0])*applied_force[1] - (
        applied_at_location[1]-input_forces_geom_center[1])*applied_force[0])


#############################
# generate nodal coordinates

coords_x = get_random_distributed_series(dx0, dx0 + lx, n_total)
coords_y = get_random_distributed_series(dy0, dy0 + ly, n_total)
coords_z = get_random_distributed_series(dz0, dz0 + lz, n_total)

if case_1:
    # CASE 1: generic
    nodal_coordinates = []
    for x, y, z in zip(coords_x, coords_y, coords_z):
        nodal_coordinates.append([x, y, z])
elif case_2a:
    # CASE 2a: testing 2d-similar case
    nodal_coordinates = []
    for x, y in zip(coords_x, coords_y):
        nodal_coordinates.append([x, y, lz/2.])
elif case_2b:
    # CASE 2b: testing 2d-similar case
    nodal_coordinates = []
    for x, z in zip(coords_x, coords_z):
        nodal_coordinates.append([x, ly/2., z])
elif case_2c:
    # CASE 2c: testing 2d-similar case
    nodal_coordinates = []
    for y, z in zip(coords_y, coords_z):
        nodal_coordinates.append([lx/2., y, z])
elif case_3:
    # CASE 3: generic with some 0.0 force and moment entries
    nodal_coordinates = []
    for x, y, z in zip(coords_x, coords_y, coords_z):
        nodal_coordinates.append([x, y, z])
elif case_4:
    # CASE 3: generic with some 0.0 force and moment entries
    nodal_coordinates = []
    for x, y, z in zip(coords_x, coords_y, coords_z):
        nodal_coordinates.append([x, y, z])

# INPUT of setup END
#############################


#############################
# COMPUTATION of setup mapping START

n_nodes = len(nodal_coordinates)
nodal_coordinates = np.asarray(nodal_coordinates)
nodes_geom_center = [np.average(nodal_coordinates[:, 0]), np.average(
    nodal_coordinates[:, 1]), np.average(nodal_coordinates[:, 2])]

# NOTE: might not be necessary to include eccentricity like this
eccentricity = [(a-b)
                for a, b in zip(input_forces_geom_center, nodes_geom_center)]
input_moments_mod = [0.0, 0.0, 0.0]
input_moments_mod[0] = input_forces_and_moments[3] + eccentricity[1] * \
    input_forces_and_moments[2] - eccentricity[2] * input_forces_and_moments[1]
input_moments_mod[1] = input_forces_and_moments[4] + eccentricity[2] * \
    input_forces_and_moments[0] - eccentricity[0] * input_forces_and_moments[2]
input_moments_mod[2] = input_forces_and_moments[5] + eccentricity[0] * \
    input_forces_and_moments[1] - eccentricity[1] * input_forces_and_moments[0]

#############################
# calculate nodal forces distributed according to the shape functions

nodal_forces = map_forces_to_nodes(
    nodal_coordinates, nodes_geom_center, input_forces_and_moments[:3] + input_moments_mod)

# COMPUTATION of setup mapping END
#############################


#############################
# CHECKS of correctness START

#############################
# perform check by calculating the recovered resultant for forces and moments from the nodal forces

recovered_forces_and_moments = [0., 0., 0., 0., 0., 0.]

for i in range(n_nodes):
    # fx, fy, fz
    fx = nodal_forces[i*3+0]
    fy = nodal_forces[i*3+1]
    fz = nodal_forces[i*3+2]

    # dx, dy, dz
    dx = nodal_coordinates[i][0] - input_forces_geom_center[0]
    dy = nodal_coordinates[i][1] - input_forces_geom_center[1]
    dz = nodal_coordinates[i][2] - input_forces_geom_center[2]

    # Fx, Fy, Fz
    recovered_forces_and_moments[0] += fx
    recovered_forces_and_moments[1] += fy
    recovered_forces_and_moments[2] += fz

    # Mx, My, Mz
    recovered_forces_and_moments[3] += dy * fz - dz * fy
    recovered_forces_and_moments[4] += dz * fx - dx * fz
    recovered_forces_and_moments[5] += dx * fy - dy * fx

#############################
# certain check prints

print("Input forces and moments: ")
print("\t" + ', '.join([str(val) for val in input_forces_and_moments]))

print("Applied at: ")
print("\t" + ', '.join([str(val) for val in input_forces_geom_center]))

print("Recovered forces and moments: ")
print("\t" + ', '.join([str(val) for val in recovered_forces_and_moments]))

print("With the geometric center of the node: ")
print("\t" + ', '.join([str(val) for val in nodes_geom_center]))

print("With the eccentricity of concentrated forces with respect to nodes center: ")
print("\t" + ', '.join([str(val) for val in eccentricity]))

print("With the difference between input and recovered values: ")
abs_diff = [abs(a-b) for a, b in zip(input_forces_and_moments,
                                     recovered_forces_and_moments)]
print("\t" + ', '.join([str(val) for val in abs_diff]))
if max(abs_diff) > 1e-3:
    raise Exception("Error too large, check code!")

print()
