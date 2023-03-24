
import numpy as np
from fem_analogy_3d_utilities import setup_fem_beam_analogy, map_forces_to_nodes, check_resultant 


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
                                        np.random.uniform(up,low),
                                        np.random.uniform(up,low)]))

# the geometric center can be a parameter to reflect the point of application of the forces
nodes_geom_center = np.array([1.25,-33.1,4.0])

# append geometric center coordinate to trigger the zero contribution element
nodal_coordinates.append(nodes_geom_center)

# the stiffness matric only depend on nodal coordinates and the center
# needs to be determined once for a geometry
stiffness_matrix = setup_fem_beam_analogy(nodal_coordinates, nodes_geom_center)

# loop over each tartget force entry
for i in range(nr_target_resultants):
    print("\nTarget force iteration: " + str(i))
    target_resultants = np.array([np.random.uniform(up,low),
                                    np.random.uniform(up,low),
                                    np.random.uniform(up,low),
                                    np.random.uniform(up,low),
                                    np.random.uniform(up,low),
                                    np.random.uniform(up,low)])

    # nodal forces need to be calculates for each given set of target resultant    
    nodal_forces, _ = map_forces_to_nodes(stiffness_matrix, target_resultants)
    perform_check = check_resultant(nodal_coordinates, nodes_geom_center, nodal_forces, target_resultants)
    print("Perform final check: ", str(perform_check))
print()