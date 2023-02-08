
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

def setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center, E=10000.0, A=100.0, I=1000.0):
   
    def get_length(p1, p2):
        # http://what-when-how.com/the-finite-element-method/fem-for-frames-finite-element-method-part-1/
        length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)**0.5
        
        return length
    
    k_total_global =np.zeros((6 + 3*len(nodal_coordinates), 6 + 3*len(nodal_coordinates)))
    
    # coord syst
    x_dir = np.array([1,0,0])
    y_dir = np.array([0,1,0])
    z_dir = np.array([0,0,1])
    # orig_syst = np.array([0,0,0])
    
    # import logging
    # logger = logging.getLogger(__name__)
    
    from commonlibs.math.vectors import unit_vector, direction_cosine#, vector_rejection
    
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

        # print(np.array_str(k_elem_local, precision=2, suppress_small=True))
        # print()
        
        # def check_symmetric(a, rtol=1e-05, atol=1e-08):
        #     return np.allclose(a, a.T, rtol=rtol, atol=atol) 
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
        
        # take global z_dir as global up 
        if abs(1 - abs(np.dot(x_elem, z_dir))) <= 1e-10:
            # up-direction and local x-axis are parallel
            # taking global y_dir as y_elem
            y_elem = unit_vector(np.copy(y_dir))
        else:
            y_elem = unit_vector(np.cross(z_dir, x_elem))
        z_elem = unit_vector(np.cross(x_elem, y_elem))
        
        # print(np.cross(x_dir, y_dir))
        # print(np.cross(y_dir, x_dir))
        # print()

        # print(direction_cosine(x_dir,x_dir))
        # print()
        
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
        # print(lx, ly, lz)
        # print(mx, my, mz)
        # print(nx, ny, nz)
        # print()
        # print(L, lx, mx, nx)
        # print(y_elem)
        # print()

        T3 = np.array([[lx, mx, nx], [ly, my, ny], [lz, mz, nz]])
        # print(np.array_str(T3, precision=2, suppress_small=True))
        # print()
        t_elem = np.zeros((9, 9))
        t_elem[0:3, 0:3] = t_elem[3:6, 3:6] = t_elem[6:9, 6:9] = T3
        # print(np.array_str(t_elem, precision=2, suppress_small=True))
        # print()

        
        k_elem_global = np.matmul(np.matmul(np.transpose(t_elem), k_elem_local), t_elem)
        # print(check_symmetric(k_elem_global))
        
        # print(np.array_str(k_elem_global, precision=2, suppress_small=True))
        # print()
        
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

        
        
    # print(np.array_str(k_total_global, precision=2, suppress_small=True))
    # print()
    
    return k_total_global

def map_forces_to_nodes(nodal_coordinates, nodes_geom_center, target_resultants):
    
    ########
    # test values - checked with Stiff3D
    #check 1
    # nodal_coordinates= [
    #     np.array([5.0,0.0,0.0]),
    #     np.array([0.0,5.0,0.0]),
    #     np.array([0.0,0.0,5.0]),
    #     np.array([5.0,5.0,5.0])]

    #check 2
    # nodal_coordinates= [
    #     np.array([5.0,0.0,0.0]),
    #     np.array([0.0,5.0,0.0]),
    #     np.array([0.0,0.0,5.0]),
    #     np.array([5.0,5.0,5.0]),
    #     np.array([-1.0,-3.0,-7.0])]
    
    # nodes_geom_center = np.array([0.0,0.0,0.0])
    # target_resultants = np.array([11.0,22.0,33.0,44.0,55.0,66.0])
    # stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center, E=1, A=1000, I=1000)

    ########
    # setup a stiffness matrix assuming the center node being connected to all other nodes
    # by a beam
    stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center)
    
    # calculate the 6 deformations - 3 translations and 3 rotation - of the center node
    # using the displacement method in 3D
    # solved by reduction, only the center node is unconstrained
    
    ########
    # solve
    center_node_deformations = np.linalg.solve(stiffness_matrix[:6,:6], target_resultants)

    print(np.array_str(stiffness_matrix[:6,:6], precision=2, suppress_small=True))
    print(np.array_str(target_resultants, precision=2, suppress_small=True))
    print(np.array_str(center_node_deformations, precision=5, suppress_small=True))
    print()
    
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

if __name__ == "__main__":
    
    nodal_coordinates= [
        np.array([5.0,0.0,0.0]),
        np.array([0.0,5.0,0.0]),
        np.array([0.0,0.0,5.0]),
        np.array([5.0,5.0,5.0]),
        np.array([-78.0,5.0,81]),
        np.array([23.0,-150.0,5.0]),
        np.array([35.0,-25.0,55.0]),
        np.array([-1.0,-3.0,-7.0])]
    
    nodal_coordinates = []
    up = 300.0
    low = -110.0
    for i in range(400):
        nodal_coordinates.append(np.array([np.random.uniform(up,low),
                                           np.random.uniform(up,low),
                                           np.random.uniform(up,low)]))
    
    nodes_geom_center = np.array([0.0,0.0,0.0])
    target_resultants = np.array([11.0,22.0,33.0,44.0,55.0,66.0])
    
    values = map_forces_to_nodes(nodal_coordinates, nodes_geom_center, target_resultants)