
import numpy as np
from fem_analogy_2d_utilities import setup_fem_beam_analogy, map_forces_to_nodes, check_resultant, ERR_ABS_TOL 

# based on the example in stiff2d_ex1.xlsm

# generating a list of nodal coordinates
nodal_coordinates = [
    np.array([20.0,60.0]),
    np.array([-60.0,60.0]),
    np.array([-20.0,-60.0]),
    np.array([-130.0,-30.0]),
    np.array([-180.0,140.0]),
]

# the geometric center can be a parameter to reflect the point of application of the forces
nodes_geom_center = np.array([-60.0,0.0])

# the stiffness matric only depend on nodal coordinates and the center
# needs to be determined once for a geometry

stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center, E=1.0, A=1000.0, I=1000.0)

target_resultants = np.array([20.0, -50.0, -50.0])

# nodal forces need to be calculates for each given set of target resultant    
nodal_forces, center_node_deformations = map_forces_to_nodes(stiffness_matrix, target_resultants)
perform_check = check_resultant(nodal_coordinates, nodes_geom_center, nodal_forces, target_resultants)
print("Perform final check: ", str(perform_check))
print()

center_node_deformations_stiff2d_reference = np.array([0.854314633420702,
                                                       -1.43822407811824,
                                                      -0.279933175826514])
residual = np.subtract(center_node_deformations, center_node_deformations_stiff2d_reference)
norm_of_residual = np.linalg.norm(residual)
print("Residual check for deformations: " , str(norm_of_residual))
if norm_of_residual > ERR_ABS_TOL:
    raise Exception("Norm of residual too large, check algorithm!")
print()