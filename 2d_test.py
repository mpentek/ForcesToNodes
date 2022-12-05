
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
    '''
    "mapping_coef_matrix" = MCM = the matrix responsible for mapping nodal forces [f_i] to resultants [F_i, M_i]
    which leads to the relation in (1):
        MCM * transpose[f_i] = transpose[F_i, M_i]

    "MCM" is known and can be constructed a-priori based on "nodal_coordinates" and "nodes_geom_center"
    "transpose[F_i, M_i]" is provided by the user as "target_resultants"

    Transforming the relation (1) into a solvable linear system with the unknown "transpose[f_i]"
    by pre-multiplying both sides with the transpose of MCM

        transpose(MCM) * MCM * transpose[f_i] = transpose(MCM) * transpose[F_i, M_i]

    Naming and re-arranging:
        LHS = transpose(MCM) * MCM
        RHS = transpose(MCM) * transpose[F_i, M_i]

    Solution for the unknown:
        tranpose[f_i] = linalg.solve(LHS, RHS)
    '''

    n_nodes = len(nodal_coordinates)

    mapping_coef_matrix = np.zeros((3, 2*n_nodes))

    # fx contribution
    mapping_coef_matrix[0, 0::2] = 1.0
    # fy contribution
    mapping_coef_matrix[1, 1::2] = 1.0
    # mz
    for i in range(0, n_nodes):
        # dx, dy
        dx = nodal_coordinates[i][0] - nodes_geom_center[0]
        dy = nodal_coordinates[i][1] - nodes_geom_center[1]

        # mz = dx * fy - dy * fx
        # fx contribution
        mapping_coef_matrix[2, 2*i+0] = -dy
        # fy contribution
        mapping_coef_matrix[2, 2*i+1] = dx

    '''
    Solving an underdetermined system, where we have in 2D 3 equations
    and the number of unknowns equals the total nodal forces, 2x number of nodes
    
    Typical problem of least squares fitting of underdetermined systems
    
    https://pages.cs.wisc.edu/~amos/412/lecture-notes/lecture17.pdf
    
    There are either infinite solutions or none
    '''

    mcm_trp = np.transpose(mapping_coef_matrix)

    # Strategy 1
    # multipliying the linear system from the left with the transposed
    # A * x = b -> A.T * A * x = A.T * b -> solve for x
    # pre-multiplying with A.T will increase the size of the system to be solved
    nodal_forces_1 = np.linalg.solve(
        np.matmul(mcm_trp, mapping_coef_matrix), np.dot(mcm_trp, target_resultants))
    # checks
    recovered_resultants_1 = np.dot(mapping_coef_matrix, nodal_forces_1)
    residual_1 = np.subtract(recovered_resultants_1, target_resultants)
    norm_of_residual_1 = np.linalg.norm(residual_1)
    print(norm_of_residual_1)

    # Strategy 2
    # substituting for a modified unknown
    # A * x = b -> A (A.T * x_mod) = b -> solve for x_mod
    # post-multiplying with A.T will decrease the size of the system to be solved
    x_mod = np.linalg.solve(
        np.matmul(mapping_coef_matrix, mcm_trp), target_resultants)
    # recover original unknown
    nodal_forces_2 = np.dot(mcm_trp, x_mod)
    # checks
    recovered_resultants_2 = np.dot(mapping_coef_matrix, nodal_forces_2)
    residual_2 = np.subtract(recovered_resultants_2, target_resultants)
    norm_of_residual_2 = np.linalg.norm(residual_2)
    print(norm_of_residual_2)

    # Strategy 3
    # let numpy solve the least square fitting problem
    nodal_forces_3 = np.linalg.lstsq(mapping_coef_matrix, target_resultants)[0]
    # checks
    recovered_resultants_3 = np.dot(mapping_coef_matrix, nodal_forces_3)
    residual_3 = np.subtract(recovered_resultants_3, target_resultants)
    norm_of_residual_3 = np.linalg.norm(residual_3)
    print(norm_of_residual_3)
    # ==>> seems to result in the same as strategy 2

    diff_12 = max(abs(np.subtract(nodal_forces_1, nodal_forces_2)))
    diff_23 = max(abs(np.subtract(nodal_forces_2, nodal_forces_3)))

    return nodal_forces_1

# OWN function definition END
#############################


#############################
# INPUT of setup START

# dimensions
lx = 32.9
ly = 64.2

# offset of starting corner
dx0 = -18.2
dy0 = -72.0

# sampling points inside the rectangle
n_total = 100

# define the input forces, moments and their point of application
# as provided by user

# create a realistic moment base on force and lever arm
applied_force = [451.3, 321.2]
applied_at_location = [-78.12, 8.33]

input_forces_geom_center = [lx/2., ly/2.]
input_forces_and_moments = []
input_forces_and_moments.extend(applied_force)
input_forces_and_moments.append((applied_at_location[1]-input_forces_geom_center[1])*applied_force[0] - (
    applied_at_location[0]-input_forces_geom_center[0])*applied_force[1])


#############################
# generate nodal coordinates

coords_x = get_random_distributed_series(dx0, dx0 + lx, n_total)
coords_y = get_random_distributed_series(dy0, dy0 + ly, n_total)

nodal_coordinates = []
for x, y in zip(coords_x, coords_y):
    nodal_coordinates.append([x, y])

# INPUT of setup END
#############################


#############################
# COMPUTATION of setup mapping START

n_nodes = len(nodal_coordinates)
nodal_coordinates = np.asarray(nodal_coordinates)
nodes_geom_center = [np.average(
    nodal_coordinates[:, 0]), np.average(nodal_coordinates[:, 1])]

# NOTE: might not be necessary to include eccentricity like this
eccentricity = [(a-b)
                for a, b in zip(input_forces_geom_center, nodes_geom_center)]
input_moments_mod = input_forces_and_moments[2] + eccentricity[0] * \
    input_forces_and_moments[1] - eccentricity[1] * input_forces_and_moments[0]

#############################
# calculate nodal forces distributed according to the shape functions

nodal_forces = map_forces_to_nodes(
    nodal_coordinates, nodes_geom_center, input_forces_and_moments[:2] + [input_moments_mod])

# COMPUTATION of setup mapping END
#############################


#############################
# CHECKS of correctness START

#############################
# perform check by calculating the recovered resultant for forces and moments from the nodal forces

recovered_forces_and_moments = [0., 0., 0.]

for i in range(n_nodes):
    # fx, fy
    fx = nodal_forces[i*2+0]
    fy = nodal_forces[i*2+1]

    # dx, dy
    dx = nodal_coordinates[i][0] - input_forces_geom_center[0]
    dy = nodal_coordinates[i][1] - input_forces_geom_center[1]

    # Fx, Fy
    recovered_forces_and_moments[0] += fx
    recovered_forces_and_moments[1] += fy

    # Mz
    recovered_forces_and_moments[2] += dx * fy - dy * fx

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
