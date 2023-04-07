import KratosMultiphysics

def Factory(params, Model):
    if(type(params) != KratosMultiphysics.Parameters):
        raise Exception(
            'expected input shall be a Parameters object, encapsulating a json string')
    return InitizalizeDistributedModelPartProcess(Model, params["Parameters"])

class InitizalizeDistributedModelPartProcess(KratosMultiphysics.Process):
    '''
    Assign level forces and recovers them (body-attached forces)
    for a model part with the appropriate 1D condition for POINT_LOAD
    split into a number of intervals
    thus the naming LevelForces

    '''
    def __init__(self, Model, params):
        KratosMultiphysics.Process.__init__(self)

        default_settings = KratosMultiphysics.Parameters("""
            {
                "model_part_name"         : "",
                "initialize_distribution" : false
            }
            """)

        params.ValidateAndAssignDefaults(default_settings)

        #############
        # on all ranks
        # could be carried out only on rank 0

        self.model_part_name = params['model_part_name'].GetString()
        self.model_part = Model[self.model_part_name]
        initialize_distribution = params['initialize_distribution'].GetBool()
        
        if initialize_distribution:
            if not(self.model_part.IsDistributed()):
                from KratosMultiphysics.mpi import DistributedModelPartInitializer
                DistributedModelPartInitializer(self.model_part, KratosMultiphysics.Testing.GetDefaultDataCommunicator(), 0).Execute()