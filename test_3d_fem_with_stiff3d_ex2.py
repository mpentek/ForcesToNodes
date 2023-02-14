
import numpy as np
from fem_analogy_3d_utilities import setup_fem_beam_analogy, map_forces_to_nodes, check_resultant 

# based on the example in stiff3d_ex1.xlsm

# generating a list of nodal coordinates
nodal_coordinates = [
    np.array([9.88,6.5,-110.0]),
    np.array([-88.5,-4.0,-3.13]),
    np.array([6.66,-98.17,14.0]),
    np.array([78.15,3.1,-45.2])
]

# the geometric center can be a parameter to reflect the point of application of the forces
nodes_geom_center = np.array([-15.73,2.47,311.0])

# the stiffness matric only depend on nodal coordinates and the center
# needs to be determined once for a geometry
stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center, E=1.0, A=125000.0, I=75000.0)

target_resultants = np.array([251.3, -22.9, 2.33, -4501.2, 98.35, -66.3])

# nodal forces need to be calculates for each given set of target resultant    
nodal_forces, center_node_deformations = map_forces_to_nodes(stiffness_matrix, target_resultants)
perform_check = check_resultant(nodal_coordinates, nodes_geom_center, nodal_forces, target_resultants)
print("Perform final check: ", str(perform_check))
print()

center_node_deformations_stiff3d_reference = np.array([5.80255483459737,
                                                       0.245957398183257,
                                                       0.165372277175854,
                                                      -1.83892555415083,
                                                       0.123857008958242,
                                                       0.646520978392889])
residual = np.subtract(center_node_deformations, center_node_deformations_stiff3d_reference)
norm_of_residual = np.linalg.norm(residual)
print("Residual check for deformations: " , str(norm_of_residual))
if norm_of_residual > 1e-4:
    raise Exception("Norm of residual too large, check algorithm!")
print()