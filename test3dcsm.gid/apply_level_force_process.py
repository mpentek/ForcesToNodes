#===============================================================================
'''
Project:Lecture - Structural Wind Engineering WS19-20
        Chair of Structural Analysis @ TUM - R. Wuchner, M. Pentek

Author: mate.pentek@tum.de

Description: Kratos level forces in body- and flow-attached coordinates

Created on:  06.01.2020
Last update: 06.01.2020
'''
#===============================================================================

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as KSM
from KratosMultiphysics.time_based_ascii_file_writer_utility import TimeBasedAsciiFileWriterUtility
import math
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
                "nr_input_intervals": 1,
                "column_ids" : [],
                "swap_sign" : false,
                "write_output_file"     : true,
                "output_file_settings"  : {}
            }
            """)

        params.ValidateAndAssignDefaults(default_settings)

        # modal part params
        self.model_part_name = params['model_part_name'].GetString()
        self.model_part = Model[self.model_part_name]
        
        input_folder = params['input_folder'].GetString()
        input_file_prefix = params['input_file_prefix'].GetString()
        input_file_extension = params['input_file_extension'].GetString()
        nr_input_intervals = params['nr_input_intervals'].GetInt()
        column_ids = params['column_ids'].GetVector()
        swap_sign = params['swap_sign'].GetBool()
        if swap_sign:
            sign_multiplier = -1
        else:
            sign_multiplier = 1

        force_labels = ['fx','fy','fz','mx','my','mz']

        if (self.model_part.GetCommunicator().MyPID() == 0):
            level_forces = {}
            accum_level_nodes = 0
            
            for l_id in range(nr_input_intervals):
                level_forces[l_id] = {}
                file_name = os.path.join(input_folder, input_file_prefix + str(l_id) + input_file_extension)
                
                # read in time series    
                level_forces[l_id]['time'] = np.loadtxt(file_name, usecols =(0,))
                # read force and moment input
                for c_id in range(column_ids.Size()):
                    level_forces[l_id][force_labels[c_id]] = np.multiply(np.loadtxt(file_name,
                                                                        usecols =(int(column_ids[c_id])),), sign_multiplier)

                # adding descriptors
                line_nr = 0
                descriptive_lines = []
                with open(file_name, 'r+') as fp:
                    while line_nr < 4:
                        line = fp.readline()
                        descriptive_lines.append(line.replace(',','').replace('\n','').split(' '))
                        line_nr += 1

                if not l_id == int(descriptive_lines[0][-1]):
                    raise Exception('Mismatch between intended and read level id!')
                level_forces[l_id]['start_id'] = [float(val) for val in descriptive_lines[1][2:]]
                level_forces[l_id]['center_id'] = [float(val) for val in descriptive_lines[2][2:]]
                level_forces[l_id]['end_id'] = [float(val) for val in descriptive_lines[3][2:]]
                
                # adding node IDs from model part
                level_forces[l_id]['node_ids'] = []
                level_forces[l_id]['node_coords'] = []
                for node in self.model_part.Nodes:
                    if level_forces[l_id]['start_id'][-1] <= node.Z0 and node.Z0 < level_forces[l_id]['end_id'][-1]:
                        level_forces[l_id]['node_ids'].append(node.Id)
                        level_forces[l_id]['node_coords'].append([node.X0, node.Y0, node.Z0])
                accum_level_nodes += len(level_forces[l_id]['node_ids'])       
                print()
                
            myval = len(self.model_part.Nodes)
            if not accum_level_nodes == len(self.model_part.Nodes):
                raise Exception('Mismatch between accumulated and total number of nodes!')
            print()
            
    def ExecuteFinalizeSolutionStep(self):

        current_time = self.model_part.ProcessInfo[KratosMultiphysics.TIME]

        if((current_time >= self.interval[0]) and (current_time < self.interval[1])):
            for idx in range(len(self.levels)):
                fb, mb = self._EvaluateLevelForces(idx)

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
                            + ' and level ' + str(idx) + '\n'
                        result_msg += ', '.join([a+b for a,
                                                 b in zip(res_labels, output_vals)])
                        self._PrintToScreen(result_msg, idx)

                    if (self.write_output_file):
                        self.levels[idx]['output_file'].write(
                            ' '.join(output_vals) + '\n')

                        # NOTE: forcing flush
                        # check in TimeBasedAsciiFileWriterUtility why this is not handled properly
                        self.levels[idx]['output_file'].flush()

    def ExecuteFinalize(self):
        '''Close output files.'''
        if (self.model_part.GetCommunicator().MyPID() == 0):
            for idx in range(len(self.levels)):
                self.levels[idx]['output_file'].close()

    def _EvaluateLevelForces(self, idx):
        # flow-attached forces: in x-y-z coordinate system
        fb = [0.0, 0.0, 0.0]
        mb = [0.0, 0.0, 0.0]

        for node_id in self.levels[idx]['node_ids']:
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

            x = node.X - self.levels[idx]['center'][0]
            y = node.Y - self.levels[idx]['center'][1]
            z = node.Z - self.levels[idx]['center'][2]
            mb[0] += y * nodal_force[2] - z * nodal_force[1]
            mb[1] += z * nodal_force[0] - x * nodal_force[2]
            mb[2] += x * nodal_force[1] - y * nodal_force[0]

        fb = self.model_part.GetCommunicator().GetDataCommunicator().SumDoubles(fb,0)
        mb = self.model_part.GetCommunicator().GetDataCommunicator().SumDoubles(mb,0)

        return fb, mb

    def _GetFileHeader(self, idx):
        header = "# Level force for level " + str(idx) + "\n"
        header += "# start: " + ', '.join(str(coord)
                                          for coord in self.levels[idx]['start']) + "\n"
        header += "# center: " + ', '.join(str(coord)
                                           for coord in self.levels[idx]['center']) + "\n"
        header += "# end: " + ', '.join(str(coord)
                                        for coord in self.levels[idx]['end']) + "\n"
        header += "# as part of " + \
            self.model_part_name + "\n"
        header += '# Time Fx\' Fy\' Fz\' Mx\' My\' Mz\'\n'

        return header

    def _PrintToScreen(self, result_msg, idx):
        KratosMultiphysics.Logger.PrintInfo(
            'ComputeLevelForceProcess', 'Level ' + str(idx) + ' - flow- and body-attached:')
        KratosMultiphysics.Logger.PrintInfo(
            'ComputeLevelForceProcess', 'Current time: ' + result_msg)
