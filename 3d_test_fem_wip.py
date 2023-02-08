
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

def setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center):
    E = 10000.0
    A = 100.0
    I = 1000.0
    
    def get_length(p1, p2):
        # http://what-when-how.com/the-finite-element-method/fem-for-frames-finite-element-method-part-1/
        length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)**0.5
        
        return length
    
    k_total_global =np.zeros((6 + 3*len(nodal_coordinates), 6 + 3*len(nodal_coordinates)))
    
    # coord syst
    x_dir = np.array([1,0,0])
    y_dir = np.array([0,1,0])
    z_dir = np.array([0,0,1])
    orig_syst = np.array([0,0,0])
    
    import logging
    logger = logging.getLogger(__name__)
    
    from commonlibs.math.vectors import unit_vector, direction_cosine, vector_rejection
    
    for idx, node in enumerate(nodal_coordinates):
        L = get_length(nodes_geom_center, node)
        
        k_u = E*A/L 
        k_vv = 3*E*I/L**3
        k_vr = 3*E*I/L**2
        k_rr = 3*E*I/L
        
        # k_elem_local = np.array([
        #     [ k_u,   0.0,   0.0, -k_u,   0.0],
        #     [ 0.0,  k_vv, -k_vr,  0.0, -k_vv],
        #     [ 0.0, -k_vr,  k_rr,  0.0,  k_vr],
        #     [-k_u,   0.0,   0.0,  k_u,   0.0],
        #     [ 0.0, -k_vv,  k_vr,  0.0,  k_vv]])
    
        k_elem_local = np.array([
            # # first nodes
            # [ k_u,   0.0,   0.0,  0.0,   0.0,   0.0,   -k_u,   0.0,   0.0],
            # [ 0.0,  k_vv,   0.0,  0.0,   0.0, -k_vr,    0.0, -k_vv,   0.0],
            # [ 0.0,   0.0,  k_vv,  0.0, -k_vr,   0.0,    0.0,   0.0, -k_vv],
            # [ 0.0,   0.0,   0.0,  0.0,   0.0,   0.0,    0.0,   0.0,   0.0],
            # [ 0.0,   0.0, -k_vr,  0.0,  k_rr,   0.0,    0.0,   0.0,  k_vr],
            # [ 0.0, -k_vr,   0.0,  0.0,   0.0,  k_rr,    0.0,  k_vr,   0.0],
            # # second node
            # [-k_u,   0.0,   0.0,  0.0,   0.0,   0.0,    k_u,   0.0,   0.0],
            # [ 0.0, -k_vv,   0.0,  0.0,   0.0,  k_vr,    0.0,  k_vv,   0.0],
            # [ 0.0,   0.0, -k_vv,  0.0,  k_vr,   0.0,    0.0,   0.0,  k_vv]])
            # first nodes
            [ k_u,   0.0,   0.0,  0.0,   0.0,   0.0,   -k_u,   0.0,   0.0],
            [ 0.0,  k_vv,   0.0,  0.0,   0.0,  k_vr,    0.0, -k_vv,   0.0],
            [ 0.0,   0.0,  k_vv,  0.0, -k_vr,   0.0,    0.0,   0.0, -k_vv],
            [ 0.0,   0.0,   0.0,  0.0,   0.0,   0.0,    0.0,   0.0,   0.0],
            [ 0.0,   0.0, -k_vr,  0.0,  k_rr,   0.0,    0.0,   0.0,  k_vr],
            [ 0.0,  k_vr,   0.0,  0.0,   0.0,  k_rr,    0.0, -k_vr,   0.0],
            # second node
            [-k_u,   0.0,   0.0,  0.0,   0.0,   0.0,    k_u,   0.0,   0.0],
            [ 0.0, -k_vv,   0.0,  0.0,   0.0, -k_vr,    0.0,  k_vv,   0.0],
            [ 0.0,   0.0, -k_vv,  0.0,  k_vr,   0.0,    0.0,   0.0,  k_vv]])
        
        def check_symmetric(a, rtol=1e-05, atol=1e-08):
            return np.allclose(a, a.T, rtol=rtol, atol=atol) 
        # print(check_symmetric(k_elem_local))
        
        # t_elem = np.array([
        #     [ c_val, -s_val, 0.0,   0.0,    0.0],
        #     [ s_val,  c_val, 0.0,   0.0,    0.0],
        #     [   0.0,   0.0,  1.0,   0.0,    0.0],
        #     [   0.0,   0.0,  0.0,  c_val, -s_val],
        #     [   0.0,   0.0,  0.0,  s_val,  c_val]])
       
        # using the syntax from here https://github.com/airinnova/framat/blob/4177a95b4ed8d95a8330365e32ca13ac9ef24640/src/framat/_element.py
        # and here https://www.engissol.com/Downloads/Technical%20Notes%20and%20examples.pdf
        x_elem = unit_vector(node - nodes_geom_center)
         
        if abs(1 - abs(np.dot(x_elem, z_dir))) <= 1e-10:
            logger.error("up-direction and local x-axis are parallel")
            raise ValueError("up-direction and local x-axis are parallel")

        z_elem = unit_vector(vector_rejection(z_dir, x_elem))
        y_elem = unit_vector(np.cross(z_elem, x_elem))
             
        # ===== Transformation matrix =====
        lx = direction_cosine(x_elem, x_dir)
        ly = direction_cosine(y_elem, x_dir)
        lz = direction_cosine(z_elem, x_dir)
        mx = direction_cosine(x_elem, y_dir)
        my = direction_cosine(y_elem, y_dir)
        mz = direction_cosine(z_elem, y_dir)
        nx = direction_cosine(x_elem, z_dir)
        ny = direction_cosine(y_elem, z_dir)
        nz = direction_cosine(z_elem, z_dir)

        T3 = np.array([[lx, mx, nx], [ly, my, ny], [lz, mz, nz]])
        # print(np.array_str(T3, precision=2, suppress_small=True))
        # print()
        t_elem = np.zeros((9, 9))
        t_elem[0:3, 0:3] = t_elem[3:6, 3:6] = t_elem[6:9, 6:9] = T3
        # print(np.array_str(t_elem, precision=2, suppress_small=True))
        # print()

        
        k_elem_global = np.matmul(np.matmul(np.transpose(t_elem), k_elem_local), t_elem)
        # print(check_symmetric(k_elem_global))
        
        # add diagonally-clustered entries corresponding to the starting node - i.e center node - of the beam
        # forces and moments
        for i in range(6):
            for j in range(6):
                k_total_global[i,j] += k_elem_global[i,j]
                
                # print(np.array_str(k_total_global[0:12,0:12], precision=2, suppress_small=True))
                # print()
        
        # add diagonally-clustered entries corresponding to the end node of of the beam
        # only forces
        for i in range(3):
            for j in range(3):
                k_total_global[6 + idx*3 + i, 6 + idx*3 + j] += k_elem_global[6+i,6+j]
                
                # print(np.array_str(k_total_global[0:12,0:12], precision=2, suppress_small=True))
                # print()
        
        # add coupling terms between nodes, which are off-diagonal
        for i in range(3):
            for j in range(6):
                # lower diagonal
                k_total_global[6 + idx*3 + i, j] += k_elem_global[6+i,j]
                # upper diagonal
                k_total_global[j, 6 + idx*3 + i] += k_elem_global[j,6+i]
                
                # print(np.array_str(k_total_global[0:12,0:12], precision=2, suppress_small=True))
                # print()
    
    return k_total_global

def map_forces_to_nodes(nodal_coordinates, nodes_geom_center, target_resultants):

    # setup a stiffness matrix assuming the center node being connected to all other nodes
    # by a beam
    stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center)
    
    # calculate the 6 deformations - 3 translations and 3 rotation - of the center node
    # using the displacement method in 3D
    # solved by reduction, only the center node is unconstrained
    center_node_deformations = np.linalg.solve(stiffness_matrix[:6,:6], target_resultants)
    
    # setup the deformation vector - apart from the center node deformations
    # all are zero translations due to the pinned support
    all_deformations = np.zeros(6+3*len(nodal_coordinates))
    # only nonzero which were previously solved for
    all_deformations[:6] = center_node_deformations
    
    # recover all forces
    all_forces = np.dot(stiffness_matrix,all_deformations)
    # first 6 are the recovered resultants
    recovered_resultants = all_forces[:6]
    # the remaining are the actual unknowns of the problem
    # return the reaction forces from each pinned node
    # flip sign for consistency
    nodal_forces = -all_forces[6:]
    
    # check target forces in the center node
    residual = np.subtract(target_resultants, recovered_resultants)
    norm_of_residual = np.linalg.norm(residual)
    print(norm_of_residual)
    if norm_of_residual > 1e-4:
        raise Exception("Norm of residual too large, check algorithm!")

    ##########################
    # check based on nodal position
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

    recovered_resultants = np.dot(mapping_coef_matrix, nodal_forces)
    residual = np.subtract(recovered_resultants, target_resultants)
    norm_of_residual = np.linalg.norm(residual)
    print(norm_of_residual)

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
# nx = 7
# ny = 9
# nz = 11

# NOTE: very interesting, 250 point with the fixed seed for the random
# generation leads to a singular matrix
n_total = 91#100 #250

input_forces_geom_center = [lx/2., ly/2., lz/2.]

# predefined setup cases
# NOTE: generic 3D does not work, special cases do work
# NOTE: choose only one of the following 4 as TRUE, preparing cases
# generic 3D -> DOES NOT WORK
case_1 = True
# various special 3D cases mimicking 2D behaviour -> WORK
case_2a = False
case_2b = False
case_2c = False
# generic case with some 0.0 force and moment entries
case_3 = False
# moments generated consistently from forces
case_4 = False

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

