
import numpy as np
from fem_analogy_3d_utilities import setup_fem_beam_analogy, map_forces_to_nodes, check_resultant 
import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as KSM


# generating nodal coordinates and target resultants
# nodal coordinates would be read in as nodes of a model part
# target resultants would be the provided concentrated forces and momoments to apply
nr_coordinates = 400
nr_target_resultants = 25 
# bounds for random generation
up = 300.0
low = -110.0

# creating mock Kratos model part
current_model = KratosMultiphysics.Model()

model_part= current_model.CreateModelPart("Main")
model_part.AddNodalSolutionStepVariable(KSM.POINT_LOAD_X)
model_part.AddNodalSolutionStepVariable(KSM.POINT_LOAD_Y)
model_part.AddNodalSolutionStepVariable(KSM.POINT_LOAD_Z)
for i in range(nr_coordinates):
    model_part.CreateNewNode(i+1,np.random.uniform(up,low),np.random.uniform(up,low),np.random.uniform(up,low))

# creating nodal coordinates for the process syntax
nodal_coordinates = []
for node in model_part.Nodes:
    nodal_coordinates.append(np.array([node.X0, node.Y0, node.Z0]))

# the geometric center can be a parameter to reflect the point of application of the forces
nodes_geom_center = np.array([1.25,-33.1,4.0])

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
    
    # assign as solution step value for the Kratos syntax
    for node in model_part.Nodes:
        node.SetSolutionStepValue(KSM.POINT_LOAD_X,nodal_forces[3*(node.Id-1)+0])
        node.SetSolutionStepValue(KSM.POINT_LOAD_Y,nodal_forces[3*(node.Id-1)+1])
        node.SetSolutionStepValue(KSM.POINT_LOAD_Z,nodal_forces[3*(node.Id-1)+2])
    
    # retrieve and check as the Kratos Syntax
    reference_point = np.copy(nodes_geom_center)
    ff = [0.0, 0.0, 0.0]
    mf = [0.0, 0.0, 0.0]

    for node in model_part.Nodes:
        # sign is flipped to go from reaction to action -> force
        nodal_force = node.GetSolutionStepValue(KSM.POINT_LOAD, 0)

        # summing up nodal contributions to get resultant for model_part
        ff[0] += nodal_force[0]
        ff[1] += nodal_force[1]
        ff[2] += nodal_force[2]

        x = node.X0 - reference_point[0]
        y = node.Y0 - reference_point[1]
        z = node.Z0 - reference_point[2]
        mf[0] += y * nodal_force[2] - z * nodal_force[1]
        mf[1] += z * nodal_force[0] - x * nodal_force[2]
        mf[2] += x * nodal_force[1] - y * nodal_force[0]
    
    total_resultants = ff+mf
    
    residual = np.subtract(total_resultants, target_resultants)
    norm_of_residual = np.linalg.norm(residual)
    print("Residual check for nodal force resultants: " , str(norm_of_residual))
    if norm_of_residual > 1e-4:
        raise Exception("Norm of residual too large, check algorithm!") 
    
print()