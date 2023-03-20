import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as KSM
from KratosMultiphysics.time_based_ascii_file_writer_utility import TimeBasedAsciiFileWriterUtility

from fem_analogy_3d_utilities import setup_fem_beam_analogy, map_forces_to_nodes, check_resultant 

import numpy as np
import os

def Factory(params, Model):
    if(type(params) != KratosMultiphysics.Parameters):
        raise Exception(
            'expected input shall be a Parameters object, encapsulating a json string')
    return ApplyLevelForceProcess(Model, params["Parameters"])

class ApplyLevelForceProcess(KratosMultiphysics.Process):
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
                "model_part_name"       : "",
                "input_folder" : "",
                "input_file_prefix" : "",
                "input_file_extension": ".dat",
                "rampup_time_upper_limit" : 1e30,
                "nr_input_intervals": 1,
                "column_ids" : [],
                "swap_sign" : false,
                "print_format"          : ".8f",
                "print_to_screen"       : true,
                "write_output_file"     : true,
                "output_file_settings"  : {}
            }
            """)

        params.ValidateAndAssignDefaults(default_settings)

        #############
        # on all ranks
        # could be carried out only on rank 0

        self.model_part_name = params['model_part_name'].GetString()
        self.model_part = Model[self.model_part_name]
        
        input_folder = params['input_folder'].GetString()
        input_file_prefix = params['input_file_prefix'].GetString()
        input_file_extension = params['input_file_extension'].GetString()
        self.nr_input_intervals = params['nr_input_intervals'].GetInt()
        
        self.time_limit = params['rampup_time_upper_limit'].GetDouble()
        
        column_ids = params['column_ids'].GetVector()
        swap_sign = params['swap_sign'].GetBool()
        if swap_sign:
            sign_multiplier = -1
        else:
            sign_multiplier = 1

        self.force_labels = ['fx','fy','fz','mx','my','mz']

        self.print_to_screen = params['print_to_screen'].GetBool()
        self.write_output_file = params['write_output_file'].GetBool()
        self.format = params["print_format"].GetString()

        #############
        # on rank 0
        if (self.model_part.GetCommunicator().MyPID() == 0):
            
            #############
            # initialize level force output
            if (self.write_output_file):

                # create/check/assign file name prefix
                output_file_name_prefix = params["model_part_name"].GetString() + "_level_force_"

                file_handler_params = KratosMultiphysics.Parameters(
                    params["output_file_settings"])

                if file_handler_params.Has("file_name"):
                    warn_msg = 'Unexpected user-specified entry found in "output_file_settings": {"file_name": '
                    warn_msg += '"' + \
                        file_handler_params["file_name"].GetString(
                        ) + '"}\n'
                    warn_msg += 'Using this specififed file name instead of the default "' + \
                        output_file_name_prefix + '"'
                    KratosMultiphysics.Logger.PrintWarning(
                        "ComputeLevelForceProcess", warn_msg)

                    output_file_name_prefix = file_handler_params["file_name"].GetString() + "_level_force_"
                else:
                    file_handler_params.AddEmptyValue("file_name")
                
            #############
            # initialize level force input
            self.level_forces = {}
            
            # NOTE: just for a check
            accum_level_nodes = 0
            
            for l_id in range(self.nr_input_intervals):
                self.level_forces[l_id] = {}
                file_name = os.path.join(input_folder, input_file_prefix + str(l_id) + input_file_extension)
                
                # read in time series    
                # NOTE: we might not need time
                self.level_forces[l_id]['time'] = np.loadtxt(file_name, usecols =(0,))
                # read force and moment input
                for c_id in range(column_ids.Size()):
                    self.level_forces[l_id][self.force_labels[c_id]] = np.multiply(np.loadtxt(file_name,
                                                                        usecols =(int(column_ids[c_id])),), sign_multiplier)

                # adding descriptors
                line_nr = 0
                descriptive_lines = []
                with open(file_name, 'r') as fp:
                    while line_nr < 4:
                        line = fp.readline()
                        descriptive_lines.append(line.replace(',','').replace('\n','').split(' '))
                        line_nr += 1

                if not l_id == int(descriptive_lines[0][-1]):
                    raise Exception('Mismatch between intended and read level id!')
                self.level_forces[l_id]['start_coords'] = [float(val) for val in descriptive_lines[1][2:]]
                self.level_forces[l_id]['center_coords'] = [float(val) for val in descriptive_lines[2][2:]]
                self.level_forces[l_id]['end_coords'] = [float(val) for val in descriptive_lines[3][2:]]
                
                ###
                # NOTE: the following block does not take care properly of MPI
                # nodes (with IDs and coordinates) are local to ranks
                # the stiffnes matrix needs to be assembled globally
                ###
                
                # adding node IDs from model part
                # by filtering Z coordinates - here along the height
                self.level_forces[l_id]['node_ids'] = []
                self.level_forces[l_id]['node_coords'] = []
                for node in self.model_part.Nodes:
                    if self.level_forces[l_id]['start_coords'][-1] <= node.Z0 and node.Z0 < self.level_forces[l_id]['end_coords'][-1]:
                        self.level_forces[l_id]['node_ids'].append(node.Id)
                        self.level_forces[l_id]['node_coords'].append([node.X0, node.Y0, node.Z0])
                accum_level_nodes += len(self.level_forces[l_id]['node_ids'])       
                #############
                
                #############
                # initialize mapping matrix using the FEM analogy
                # NOTE: the geometric center might be different
                self.level_forces[l_id]['stiffness_matrix'] = setup_fem_beam_analogy(self.level_forces[l_id]['node_coords'], 
                                                                                self.level_forces[l_id]['center_coords'])
            

                #############
                # initialize level force output
                if (self.write_output_file):
                    # file for each level
                    file_handler_params["file_name"].SetString(
                            output_file_name_prefix + str(l_id) + '.dat')
                    file_header = self._GetFileHeader(l_id)

                    self.level_forces[l_id]['output_file'] = TimeBasedAsciiFileWriterUtility(self.model_part,
                                                                                        file_handler_params, file_header).file

            
            # NOTE: for an internal check  
            if not accum_level_nodes == len(self.model_part.Nodes):
                raise Exception('Mismatch between accumulated and total number of nodes!')

    def ExecuteInitializeSolutionStep(self):
        
        # NOTE: check if whe should base it on time
        # linear time ramp-up
        current_time = self.model_part.ProcessInfo[KratosMultiphysics.TIME]
        time_fctr = 1.0
        if current_time <= self.time_limit:
            time_factor = current_time/self.time_limit
            
        step = self.model_part.ProcessInfo[KratosMultiphysics.STEP]
        
        #############
        # on rank 0
        if (self.model_part.GetCommunicator().MyPID() == 0):
            for l_id in range(self.nr_input_intervals):
            
                #############
                # initialize level force input
                # NOTE: we might want to use a ramp-up function of the force input
                target_resultants = np.zeros(len(self.force_labels))
                for c, f_l in enumerate(self.force_labels):
                    target_resultants[c] = time_fctr * self.level_forces[l_id][f_l][step]
                
                # nodal forces need to be calculates for each given set of target resultant    
                nodal_forces, _ = map_forces_to_nodes(self.level_forces[l_id]['stiffness_matrix'], target_resultants)
                perform_check = check_resultant(self.level_forces[l_id]['node_coords'], self.level_forces[l_id]['center_coords'], nodal_forces, target_resultants)
                print("Perform final check: ", str(perform_check))
            
                #############
                # apply as in Kratos
                
                ###
                # NOTE: the following block does not take care properly of MPI
                # nodal forces can be calculated glocally on rank 0
                # the result and distribution needs to happen for each node on respective ranks
                # so local node IDs
                ###
                
                nodal_point_load_val = KratosMultiphysics.Vector(3)            
                for c, node_id in enumerate(self.level_forces[l_id]['node_ids']):
                    for i in range(nodal_point_load_val.Size()):
                        nodal_point_load_val[i] = nodal_forces[c*3 + i]
                    # for StructuralMechanics: each node has a corresponding 1D condition, which has POINT_LOAD as an assigned value
                    self.model_part.Conditions[node_id].SetValue(KSM.POINT_LOAD, nodal_point_load_val)
            
    def ExecuteFinalizeSolutionStep(self):

        current_time = self.model_part.ProcessInfo[KratosMultiphysics.TIME]

        for l_id in range(self.nr_input_intervals):
            fb, mb = self._EvaluateLevelForces(l_id)

            if (self.model_part.GetCommunicator().MyPID() == 0):
                output = []
                output.extend(fb)
                output.extend(mb)

                output_vals = [format(val, self.format) for val in output]
                # not formatting time in order to not lead to problems with time recognition
                # in the file writer when restarting
                output_vals.insert(0, str(current_time))

                res_labels = ['time: ',
                                'fx\': ', 'fy\': ', 'fz\': ', 'mx\': ', 'my\': ', 'mz\': ']

                if (self.print_to_screen):
                    result_msg = 'Level force evaluation for model part ' + \
                        self.model_part_name + '\n' \
                        + ' and level ' + str(l_id) + '\n'
                    result_msg += ', '.join([a+b for a,
                                                b in zip(res_labels, output_vals)])
                    self._PrintToScreen(result_msg, l_id)

                    print()
                if (self.write_output_file):
                    self.level_forces[l_id]['output_file'].write(
                        ' '.join(output_vals) + '\n')

                    # NOTE: forcing flush
                    # check in TimeBasedAsciiFileWriterUtility why this is not handled properly
                    self.level_forces[l_id]['output_file'].flush()

    def ExecuteFinalize(self):
        '''Close output files.'''
        if (self.model_part.GetCommunicator().MyPID() == 0):
            for l_id in range(len(self.level_forces)):
                self.level_forces[l_id]['output_file'].close()

    def _EvaluateLevelForces(self, l_id):
        # flow-attached forces: in x-y-z coordinate system
        fb = [0.0, 0.0, 0.0]
        mb = [0.0, 0.0, 0.0]

        for node_id in self.level_forces[l_id]['node_ids']:
            node = self.model_part.Nodes[node_id]

            # for FluidDynamics: each node has REACTION as a solution step value
            # sign is flipped to go from reaction to action -> force
            # nodal_force = (-1) * node.GetSolutionStepValue(KratosMultiphysics.REACION, 0)
            
            # for StructuralMechanics: each node has a corresponding 1D condition, which has POINT_LOAD as an assigned value
            nodal_force = self.model_part.Conditions[node_id].GetValue(KSM.POINT_LOAD)

            # summing up nodal contributions to get resultant for model_part
            fb[0] += nodal_force[0]
            fb[1] += nodal_force[1]
            fb[2] += nodal_force[2]

            x = node.X - self.level_forces[l_id]['center_coords'][0]
            y = node.Y - self.level_forces[l_id]['center_coords'][1]
            z = node.Z - self.level_forces[l_id]['center_coords'][2]
            mb[0] += y * nodal_force[2] - z * nodal_force[1]
            mb[1] += z * nodal_force[0] - x * nodal_force[2]
            mb[2] += x * nodal_force[1] - y * nodal_force[0]

        fb = self.model_part.GetCommunicator().GetDataCommunicator().SumDoubles(fb,0)
        mb = self.model_part.GetCommunicator().GetDataCommunicator().SumDoubles(mb,0)

        return fb, mb

    def _GetFileHeader(self, l_id):
        header = "# Level force for level " + str(l_id) + "\n"
        header += "# start: " + ', '.join(str(coord)
                                          for coord in self.level_forces[l_id]['start_coords']) + "\n"
        header += "# center: " + ', '.join(str(coord)
                                           for coord in self.level_forces[l_id]['center_coords']) + "\n"
        header += "# end: " + ', '.join(str(coord)
                                        for coord in self.level_forces[l_id]['end_coords']) + "\n"
        header += "# as part of " + \
            self.model_part_name + "\n"
        header += '# Time Fx\' Fy\' Fz\' Mx\' My\' Mz\'\n'

        return header

    def _PrintToScreen(self, result_msg, l_id):
        KratosMultiphysics.Logger.PrintInfo(
            'ComputeLevelForceProcess', 'Level ' + str(l_id) + ' - flow- and body-attached:')
        KratosMultiphysics.Logger.PrintInfo(
            'ComputeLevelForceProcess', 'Current time: ' + result_msg)
