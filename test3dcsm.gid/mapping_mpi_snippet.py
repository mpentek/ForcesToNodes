# follow applications/CoSimulationApplication/python_scripts/convergence_accelerators/convergence_accelerator_wrapper.py
# https://github.com/KratosMultiphysics/Kratos/blob/master/applications/CoSimulationApplication/python_scripts/convergence_accelerators/convergence_accelerator_wrapper.py


# local search
# => find out how many nodes 
rank = 0
data_comm = model_part.GetCommunicator().GetDataCommunicator()

participating_nodes = [n for n in model_part.Nodes if (n.Z > 0 and n.Z < 100)]

sizes_from_ranks = np.cumsum(data_comm.GatherInts([len(participating_nodes)], rank))

local_coords = [n.X for n in participating_nodes]

global_coords = np.array(np.concatenate(data_comm.GathervDoubles(local_coords, rank)))

if data_comm.Rank() == rank:
    # compute stiffness_matrix
    matrix = ...
    

for n in time_steps:
    if data_comm.Rank() == rank:
        # read stuff from file
        ext_loads = np.loadtxt(...)
        
        # use striffness_matrix to compute global_loads
        global_loads = matrix x ext_loads
    
        data_to_scatter = np.split(global_loads, sizes_from_ranks[:-1])
    else:
        data_to_scatter = []
    
    local_loads = data_comm.ScattervDoubles(data_to_scatter, rank)
    
    assert len(local_loads) == len(participating_nodes)
    
    for l, n in zip(local_loads, participating_nodes):
        n.SetSolutionStepValue(l)
    

