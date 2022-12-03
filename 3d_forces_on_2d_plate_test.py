
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

    mcm_trp = np.transpose(mapping_coef_matrix)
    LHS = np.matmul(mcm_trp, mapping_coef_matrix)

    RHS = np.dot(mcm_trp, target_resultants)
    nodal_forces = np.linalg.solve(LHS, RHS)

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
# NOTE: no offset
dx0 = 0.0#-18.2
dy0 = 0.0#-72.0
dz0 = 0.0#23.3

# sampling points inside the rectangle
nx = 7
ny = 9
nz = 11
n_total = 50  # 75 #200 #max(nx, ny, nz)**3

# define the input forces, moments and their point of application
input_forces_and_moments = [451.3, 321.2, -512.7, -1545.2, -2117.8, 3001.3]
    
input_forces_geom_center = [lx/2., ly/2., lz/2.]

#############################
# generate nodal coordinates

coords_x = get_random_distributed_series(dx0, dx0 + lx, n_total)
coords_y = get_random_distributed_series(dy0, dy0 + ly, n_total)
coords_z = get_random_distributed_series(dz0, dz0 + lz, n_total)

nodal_coordinates = []
for x, y in zip(coords_x, coords_y):
    nodal_coordinates.append([x, y, lz/2.])

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
