
import numpy as np
from fem_analogy_3d_utilities import setup_fem_beam_analogy, map_forces_to_nodes, check_resultant 

# based on the example in stiff3d_ex1.xlsm

# generating a list of nodal coordinates
nodal_coordinates = [
    np.array([3.0,0.0,0.0]),
    np.array([0.0,4.0,0.0]),
    np.array([0.0,0.0,5.0]),
    np.array([3.0,4.0,5.0])
]

# the geometric center can be a parameter to reflect the point of application of the forces
nodes_geom_center = np.array([0.0,0.0,0.0])

# the stiffness matric only depend on nodal coordinates and the center
# needs to be determined once for a geometry
stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center, E=1.0, A=1000.0, I=1000.0)

target_resultants = np.array([11.0, -22.0, 33.0, -44.0, 55.0, -66.0])

# nodal forces need to be calculates for each given set of target resultant    
nodal_forces, center_node_deformations = map_forces_to_nodes(stiffness_matrix, target_resultants)
perform_check = check_resultant(nodal_coordinates, nodes_geom_center, nodal_forces, target_resultants)
print("Perform final check: ", str(perform_check))
print()

center_node_deformations_stiff3d_reference = np.array([-0.015678705645457,
                                                       -0.0706822817769753,
                                                        0.159219536145635,
                                                       -0.0516533236593279,
                                                        0.0560509531234531,
                                                       -0.0209843702516317])
residual = np.subtract(center_node_deformations, center_node_deformations_stiff3d_reference)
norm_of_residual = np.linalg.norm(residual)
print("Residual check for deformations: " , str(norm_of_residual))
if norm_of_residual > 1e-4:
    raise Exception("Norm of residual too large, check algorithm!")
print()