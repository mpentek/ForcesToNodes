
import numpy as np

#############################
# OWN function definition START

DIMENSION = 3
DOFS_PER_NODE = 6
ERR_ABS_TOL = 1e-5
ERR_REL_TOL = 1e-10

def setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center, E=10000.0, A=100.0, I=1000.0):
   
    # internal definitions
    def get_unit_vector(v):
        return v / np.linalg.norm(v)

    def get_direction_cosine(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_length(p1, p2):       
        return np.linalg.norm(np.subtract(p2, p1))
    
    # initial checks
    if len(nodal_coordinates) == 0:
        raise Exception("nodal_coordinates has no elements, check algorithm or setup!")
    elif len(nodal_coordinates) == 1:
        if get_length(nodal_coordinates[0], nodes_geom_center) < ERR_ABS_TOL:
            raise Exception("Only one node in nodal_coordinates and this has the same location as nodes_geom_center, check algorithm or setup!")
        else:
            raise Warning("Only one node in nodal_coordinates!")
    
    # main algorithm
    k_total_global =np.zeros((DOFS_PER_NODE + DIMENSION*len(nodal_coordinates), DOFS_PER_NODE + DIMENSION*len(nodal_coordinates)))
    
    # coord syst
    # unit direction vectors
    x_dir = np.array([1,0,0])
    y_dir = np.array([0,1,0])
    z_dir = np.array([0,0,1])
    # orig_syst = np.array([0,0,0])
    
    for idx, node in enumerate(nodal_coordinates):
                
        L = get_length(nodes_geom_center, node)
        if not L < ERR_ABS_TOL:
            
            k_u = E*A/L 
            k_vv = 3*E*I/L**3
            k_vr = 3*E*I/L**2
            k_rr = 3*E*I/L
            
            k_elem_local = np.array([
                # first node - displacements and rotations
                [ k_u,   0.0,   0.0,  0.0,   0.0,   0.0, -k_u,   0.0,   0.0],
                [ 0.0,  k_vv,   0.0,  0.0,   0.0,  k_vr,  0.0, -k_vv,   0.0],
                [ 0.0,   0.0,  k_vv,  0.0, -k_vr,   0.0,  0.0,   0.0, -k_vv],
                [ 0.0,   0.0,   0.0,  0.0,   0.0,   0.0,  0.0,   0.0,   0.0],
                [ 0.0,   0.0, -k_vr,  0.0,  k_rr,   0.0,  0.0,   0.0,  k_vr],
                [ 0.0,  k_vr,   0.0,  0.0,   0.0,  k_rr,  0.0, -k_vr,   0.0],
                # second node - only displacements
                [-k_u,   0.0,   0.0,  0.0,   0.0,   0.0,  k_u,   0.0,   0.0],
                [ 0.0, -k_vv,   0.0,  0.0,   0.0, -k_vr,  0.0,  k_vv,   0.0],
                [ 0.0,   0.0, -k_vv,  0.0,  k_vr,   0.0,  0.0,   0.0,  k_vv]])
        
            # using the syntax from here https://github.com/airinnova/framat/blob/4177a95b4ed8d95a8330365e32ca13ac9ef24640/src/framat/_element.py
            # and here https://www.engissol.com/Downloads/Technical%20Notes%20and%20examples.pdf
            x_elem = get_unit_vector(np.subtract(node, nodes_geom_center))
            
            # take global z_dir as global up 
            if abs(1 - abs(np.dot(x_elem, z_dir))) <= ERR_ABS_TOL:
                # up-direction and local x-axis are parallel
                # taking global y_dir as y_elem
                y_elem = get_unit_vector(np.copy(y_dir))
            else:
                y_elem = get_unit_vector(np.cross(z_dir, x_elem))
            z_elem = get_unit_vector(np.cross(x_elem, y_elem))
                    
            #######
            lx = get_direction_cosine(x_elem, x_dir)
            ly = get_direction_cosine(y_elem, x_dir)
            lz = get_direction_cosine(z_elem, x_dir)
            mx = get_direction_cosine(x_elem, y_dir)
            my = get_direction_cosine(y_elem, y_dir)
            mz = get_direction_cosine(z_elem, y_dir)
            nx = get_direction_cosine(x_elem, z_dir)
            ny = get_direction_cosine(y_elem, z_dir)
            nz = get_direction_cosine(z_elem, z_dir)

            T3 = np.array([[lx, mx, nx], [ly, my, ny], [lz, mz, nz]])
            t_elem = np.zeros((DIMENSION**2, DIMENSION**2))
            t_elem[0:DIMENSION, 0:DIMENSION] = t_elem[DIMENSION:2*DIMENSION, DIMENSION:2*DIMENSION] = t_elem[2*DIMENSION:3*DIMENSION, 2*DIMENSION:3*DIMENSION] = T3
            
            k_elem_global = np.matmul(np.matmul(np.transpose(t_elem), k_elem_local), t_elem)
            
        else:
            # node too close to center node
            msg = "Node center :" + ', '.join(str(val) for val in nodes_geom_center)
            msg += " too close to considered node: " + ', '.join(str(val) for val in node)
            msg += " with distance: " + str(L)
            msg += ". Adding with zero contribution."
            print(msg)        
            
            # adding element stiffness matrix to result in zero contributions
            k_elem_global = np.zeros([DOFS_PER_NODE*2-3,DOFS_PER_NODE*2-3])
        
        # add diagonally-clustered entries corresponding to the starting node - i.e center node - of the beam
        # forces and moments
        for i in range(DOFS_PER_NODE):
            for j in range(DOFS_PER_NODE):
                k_total_global[i,j] += k_elem_global[i,j]
                
        # add diagonally-clustered entries corresponding to the end node of of the beam
        for i in range(DIMENSION):
            for j in range(DIMENSION):
                k_total_global[DOFS_PER_NODE + idx*DIMENSION + i, DOFS_PER_NODE + idx*DIMENSION + j] += k_elem_global[DOFS_PER_NODE+i,DOFS_PER_NODE+j]
                
        # add coupling terms between nodes, which are off-diagonal
        for i in range(DIMENSION):
            for j in range(DOFS_PER_NODE):
                # lower diagonal
                k_total_global[DOFS_PER_NODE + idx*DIMENSION + i, j] += k_elem_global[DOFS_PER_NODE+i,j]
                # upper diagonal
                k_total_global[j, DOFS_PER_NODE + idx*DIMENSION + i] += k_elem_global[j,DOFS_PER_NODE+i]

    # final checks
    if np.isnan(k_total_global).any():
        raise Exception("NaN in k_total_global, check algorithm or setup!")
        
    return k_total_global

def map_forces_to_nodes(stiffness_matrix, target_resultants):
    
    ########
    # setup a stiffness matrix assuming the center node being connected to all other nodes
    # by a beam

    # calculate the 6 deformations - 3 translations and 3 rotation - of the center node
    # using the displacement method in 3D
    # solved by reduction, only the center node is unconstrained
    
    ########
    # solve   
    center_node_deformations = np.linalg.solve(stiffness_matrix[:DOFS_PER_NODE,:DOFS_PER_NODE], target_resultants)
    
    # setup the deformation vector - apart from the center node deformations
    # all are zero translations due to the pinned support
    all_deformations = np.zeros(stiffness_matrix.shape[0])
    # only nonzero which were previously solved for
    all_deformations[:DOFS_PER_NODE] = center_node_deformations
    
    # recover all forces
    all_forces = np.dot(stiffness_matrix,all_deformations)
    # first 6 are the recovered resultants
    recovered_resultants = all_forces[:DOFS_PER_NODE]
    # the remaining are the actual unknowns of the problem
    # return the reaction forces from each pinned node
    # flip sign for consistency
    nodal_forces = -all_forces[DOFS_PER_NODE:]
    
    # check target forces in the center node
    residual = np.subtract(target_resultants, recovered_resultants)   
    abs_norm_of_residual = np.linalg.norm(residual)
    rel_norm_of_residual = abs_norm_of_residual/np.linalg.norm(target_resultants)
    msg = "Residual check in map_forces_to_nodes"
    msg += "\n\tabsolute residual: " + str(abs_norm_of_residual)
    msg += "\n\trelative residual: " + str(rel_norm_of_residual)
    print(msg)
    if (abs_norm_of_residual > ERR_ABS_TOL or rel_norm_of_residual > ERR_REL_TOL):
        raise Exception("Norm of residual too large, check algorithm!")

    return nodal_forces, center_node_deformations

def check_resultant(nodal_coordinates, nodes_geom_center, nodal_forces, target_resultants):

    ##########################
    # check based on nodal position
    n_nodes = len(nodal_coordinates)

    mapping_coef_matrix = np.zeros((DOFS_PER_NODE, DIMENSION*n_nodes))

    # fx contribution
    mapping_coef_matrix[0, 0::DIMENSION] = 1.0
    # fy contribution
    mapping_coef_matrix[1, 1::DIMENSION] = 1.0
    # fz contribution
    mapping_coef_matrix[2, 2::DIMENSION] = 1.0
    # mx, my, mz
    for i in range(0, n_nodes):
        # dx, dy, dz
        dx = nodal_coordinates[i][0] - nodes_geom_center[0]
        dy = nodal_coordinates[i][1] - nodes_geom_center[1]
        dz = nodal_coordinates[i][2] - nodes_geom_center[2]

        # mx = dy * fz - dz * fy
        # fy contribution
        mapping_coef_matrix[3, DIMENSION*i+1] = -dz
        # fz contribution
        mapping_coef_matrix[3, DIMENSION*i+2] = dy

        # my = dz * fx - dx * fz
        # fx contribution
        mapping_coef_matrix[4, DIMENSION*i+0] = dz
        # fz contribution
        mapping_coef_matrix[4, DIMENSION*i+2] = -dx

        # mz = dx * fy - dy * fx
        # fx contribution
        mapping_coef_matrix[5, DIMENSION*i+0] = -dy
        # fy contribution
        mapping_coef_matrix[5, DIMENSION*i+1] = dx

    recovered_resultants = np.dot(mapping_coef_matrix, nodal_forces)
    residual = np.subtract(recovered_resultants, target_resultants)
    abs_norm_of_residual = np.linalg.norm(residual)
    rel_norm_of_residual = abs_norm_of_residual/np.linalg.norm(target_resultants)
    msg = "Residual check in check_resultants"
    msg += "\n\tabsolute residual: " + str(abs_norm_of_residual)
    msg += "\n\trelative residual: " + str(rel_norm_of_residual)
    print(msg)
    if (abs_norm_of_residual > ERR_ABS_TOL or rel_norm_of_residual > ERR_REL_TOL):
        raise Exception("Norm of residual too large, check algorithm!")

    return not((abs_norm_of_residual > ERR_ABS_TOL) or (rel_norm_of_residual > ERR_REL_TOL))