
import numpy as np

#############################
# OWN function definition START

DIMENSION = 2
DOFS_PER_NODE = 3
ERR_ABS_TOL = 1e-5
ERR_REL_TOL = 1e-10

def setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center, E=10000.0, A=100.0, I=1000.0):
    
    def get_length_and_angle_coefs(p1, p2):
        # http://what-when-how.com/the-finite-element-method/fem-for-frames-finite-element-method-part-1/
        length = np.linalg.norm(np.subtract(p2, p1))
        cos_val = (p2[0]-p1[0]) / length
        sin_val = (p2[1]-p1[1]) / length
        
        return length, cos_val, sin_val
    
    k_total_global =np.zeros((DOFS_PER_NODE + DIMENSION*len(nodal_coordinates), DOFS_PER_NODE + DIMENSION*len(nodal_coordinates)))
    
    for idx, node in enumerate(nodal_coordinates):
        L, c_val, s_val = get_length_and_angle_coefs(nodes_geom_center, node)
        
        if not L < ERR_ABS_TOL:
            
            k_u = E*A/L 
            k_vv = 3*E*I/L**3
            k_vr = 3*E*I/L**2
            k_rr = 3*E*I/L
            
            k_elem_local = np.array([
                # first node - displacement and rotation
                [ k_u,   0.0,   0.0, -k_u,    0.0],
                [ 0.0,  k_vv,   k_vr,  0.0, -k_vv],
                [ 0.0,  k_vr,   k_rr,  0.0, -k_vr],
                # second node - only displacements
                [-k_u,   0.0,   0.0,  k_u,    0.0],
                [ 0.0, -k_vv,  -k_vr,  0.0,  k_vv]])
            
            t_elem = np.array([
                [  c_val, s_val, 0.0,   0.0,    0.0],
                [ -s_val, c_val, 0.0,   0.0,    0.0],
                [    0.0,   0.0, 1.0,   0.0,    0.0],
                [    0.0,   0.0, 0.0,  c_val, s_val],
                [    0.0,   0.0, 0.0, -s_val, c_val]])
            
            k_elem_global = np.matmul(np.matmul(np.transpose(t_elem), k_elem_local), t_elem)
            
        else:
            # node too close to center node
            msg = "Node center :" + ','.join(str(val) for val in nodes_geom_center)
            msg += " too close to considered node: " + ','.join(str(val) for val in node)
            msg += " with distance: " + str(L)
            msg += ". Adding with zero contribution."
            print(msg)        
            
            # adding element stiffness matrix to result in zero contributions
            k_elem_global = np.zeros([DOFS_PER_NODE*2-1,DOFS_PER_NODE*2-1])
        
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

    if np.isnan(k_total_global).any():
        raise Exception("NaN in k_total_global, check algorithm!")
    
    return k_total_global

def map_forces_to_nodes(stiffness_matrix, target_resultants):
    
    ########
    # setup a stiffness matrix assuming the center node being connected to all other nodes
    # by a beam

    # calculate the 3 deformations - 2 translations and 1 rotation - of the center node
    # using the displacement method in 2D
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
    # first 3 are the recovered resultants
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
    # mz
    for i in range(0, n_nodes):
        # dx, dy
        dx = nodal_coordinates[i][0] - nodes_geom_center[0]
        dy = nodal_coordinates[i][1] - nodes_geom_center[1]

        # mz = dx * fy - dy * fx
        # fx contribution
        mapping_coef_matrix[2, DIMENSION*i+0] = -dy
        # fy contribution
        mapping_coef_matrix[2, DIMENSION*i+1] = dx

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

if __name__ == "__main__":
    # generating nodal coordinates and target resultants
    # nodal coordinates would be read in as nodes of a model part
    # target resultants would be the provided concentrated forces and momoments to apply
    nr_coordinates = 400
    nr_target_resultants = 25 

    # generating a list of nodal coordinates
    nodal_coordinates = []
    up = 300.0
    low = -110.0
    for i in range(nr_coordinates):
        nodal_coordinates.append(np.array([np.random.uniform(up,low),
                                            np.random.uniform(up,low)]))

    # the geometric center can be a parameter to reflect the point of application of the forces
    nodes_geom_center = np.array([1.25,-33.1])

    # the stiffness matric only depend on nodal coordinates and the center
    # needs to be determined once for a geometry
    stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center)

    # loop over each tartget force entry
    for i in range(nr_target_resultants):
        print("\nTarget force iteration: " + str(i))
        target_resultants = np.array([np.random.uniform(up,low),
                                        np.random.uniform(up,low),
                                        np.random.uniform(up,low)])

        # nodal forces need to be calculates for each given set of target resultant    
        nodal_forces, _ = map_forces_to_nodes(stiffness_matrix, target_resultants)
        perform_check = check_resultant(nodal_coordinates, nodes_geom_center, nodal_forces, target_resultants)
        print("Perform final check: ", str(perform_check))
    print()