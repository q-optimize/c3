from zrl.oc_backend import *
from io import StringIO
import datetime
import json
import numpy as np
import os
import shutil
import subprocess
import yaml
import copy

# IBM imports
from ibmqc.pulse_gen import qasm2sq2
from ibmqc.file_utils import import_yaml, search_path
from ibmqc.util import execute
from ibmqc.experiments.sqore.expt_loader import load_experiment

from zrl import data_loading as zrl_dl

import ibmqc.experiments.sqore.expt_fitter as ef
import ibmqc.prefs as prefs
import matplotlib.pyplot as plt

def get_current_iter_dir(base_dir, iter_name, data_dir):
   runs = os.listdir(base_dir + data_dir)
   for run in runs:
       if iter_name in run:
           return data_dir + '/' + run + '/'

def make_template_sqore_file(sequences, parameter_batch, sqore_dir):
    """
    For a single set of pulse parameters we run rep different RB
    sequences of some length.
    :param sequences: List of sqore objects containing RB sequences
    :param parameter_batch: list of dictionairies of pulse parameters and values.
    :param sqore_dir: the directory where the sqore file will be saved.
    """
    # prepare the calibration string

    final_string = ''
    
    # insert parameter solutions:

    include_cals = True
    for ii in range(len(sequences)):
        qasm = sequences[ii]
        print(qasm)
        sqore = build_sqore(qasm, 0)
        sqore.print_sq2()
        # extract string from sqore obj to change it
        sqore_string = StringIO()
        sqore.print_sq2(sqore_string)
        sqore_lines = sqore_string.getvalue().split('\n')

        if include_cals:
            cal_sqore = build_sqore(qasm, 3)

            # extract string from sqore obj to change it
            cal_sqore_string = StringIO()
            cal_sqore.print_sq2(cal_sqore_string)
            cal_sqore_lines = cal_sqore_string.getvalue().split('\n')

        new_sqore_string = ''

        for jj in range(len(parameter_batch)):

            if include_cals:
                lines = cal_sqore_lines
                include_cals = False
            else:
                lines = sqore_lines


            param_dict = parameter_batch[jj]
            new_sqore_lines = []

            for line in lines:
                seq_elements = line.split(';')
                ch = seq_elements[0]

                new_seq = [ch]

                for gate in seq_elements[1:]:
                    gate = gate.replace(' ', '')

                    parameters = get_parameters(ch, gate, param_dict)

                    new_seq.append(gate + parameters)

                new_line = '; '.join(new_seq)

                new_line = new_line.replace('xval(0,{})'.format(self.rb_length), 'xval({},{})'.format(ii, jj))

                new_sqore_lines.append(new_line)
            new_sqore_string += '\n'.join(new_sqore_lines)
        final_string += new_sqore_string+'\n'

    sq2file = open(sqore_dir+"CMASearch.sq2", "w")
    sq2file.write(final_string)
    sq2file.close()


def get_parameters(channel, gate, param_dict):
    """
    :param channel: the logical_channel, i.e. firs tag of a sqore file.
    :param gate: the name of the gate in gatedef.
    :param param_dict: a dictionary of pulse parameters.
    :return: A string of gate parameters e.g. ('paramA': 0.1).
    """
    if channel not in param_dict:
        return ''

    if gate not in param_dict[channel]:
        return ''

    parameters = param_dict[channel][gate]

    if 'mapto' in parameters:
        mapto_gate = parameters['mapto']

        mapto_parameters = param_dict[channel][mapto_gate]

        for p in mapto_parameters:
            if p not in parameters:
                parameters[p] = mapto_parameters[p]

        del parameters['mapto']

    temp = ['\''+p+'\': '+str(parameters[p]) for p in parameters]

    # Avoid returning '()'
    if len(temp) == 0:
        return ''

    return '(' + ', '.join(temp) + ')'


def build_sqore(qasm, num_cals, exp_qasm=None):
    """
    This function generates a sqore object from a qasm string
    :param qasm: qasm string
    :param exp_qasm:
    :returns sqore object
    """
    gate_def = '/home/measure/analysis/ZRLLabScripts/jupyter_scripts/gatedef_OC.txt'
    gdpath = ['.', os.path.dirname(qasm2sq2.__file__)]
    gate_def = search_path(prefs.get('gatedefpath', gdpath),gate_def)

    if not exp_qasm:
        exp_qasm = {}

    sqore = qasm2sq2.translate_qasm(qasm2sq2.read_gatedef(gate_def),
        fstr=qasm,
        default=exp_qasm.get('default', 'QId'),
        trigger=exp_qasm.get('trigger', True),
        measure=exp_qasm.get('measure', True),
        cals2N=exp_qasm.get('cals2N', False),
        cals=exp_qasm.get('cals', num_cals),
        relative=exp_qasm.get('relative', False),
        emptyseq=exp_qasm.get('emptyseq',None))
    # Document .qasm file
    return sqore




class ExpCMARunner(CMARunner):
    """
    Class to acquirer and manage data during pulse optimization.
    """

    def __init__(self, dir_name, base_dir, sqore_dir, ch='M1', run_sqore='run_sqore'):
        CMARunner.__init__(self)
        self.channel = ch
        self.sqore_dir = sqore_dir
        self.dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")+'_'+dir_name
        self.run_sqore = run_sqore
        self.base_dir = base_dir
        self.rb_length = 0

    def evaluate_sequences(self, sequences, solutions, batch_num=1, rounds=200):
        """
        When running on the quantum hardware the following steps are done
        1) Translate parameters to a batch of sqore sequences.
        2) Evaluate a batch.
        3) Extract fidelity of each sequence.
        4) Repeat from 1) for next batch.

        :param rep: number of different RB sequences to run for the same parameters.
        :param rb_len: length of the RB sequences as number of Cliffords.
        :param solutions: contains the part of the batche that should be evaluated.
        """

        self.make_template_sqore_file(sequences, solutions, self.sqore_dir)

        name_tag = 'CMASearchIter{}Batch{}'.format(self.iter_num, batch_num) 

        cmd = self.run_sqore + ' -b --dir ' + self.dir_name + ' --name '
        cmd += name_tag + ' --set rounds={}'.format(rounds)+' OCRB.yaml'
        
        ret_val = -1

        base_dir = os.getcwd()+'//'

        # Run until we get data
        while ret_val != 0:
            ret_val = execute.run_warn(cmd)

            # delete the dir if the run fails and measure again.
            if ret_val != 0:
                shutil.rmtree(base_dir + zrl_dl.get_current_iter_dir(base_dir, name_tag, self.dir_name))

        data_dir = base_dir + zrl_dl.get_current_iter_dir(base_dir, name_tag, self.dir_name)

        self.iter_num += 1

        rb_avg = self.load_rb_data(data_dir, self.channel)

        # Save status to history and write to files.
        self.save_history(rb_avg, solutions)

        self.update_best_ever(rb_avg, solutions)

        return rb_avg

    @staticmethod
    def load_rb_data(data_dir, ch):
        """
        The RB curve in qc_code follows Ap**m + B where the error per
        Clifford r has a behaviour similar to 1 - p.
        Note: in Kelly PRL (2018) single qubit RB follows p = 1-2r.

        :param data_dir: directory from which to load the data.
        :param ch: channel e.g. 'M1'
        """

        # Load the data
        ld = load_experiment(data_dir, )

        with open(data_dir + 'exp_params.yaml', 'r') as f:
            exp_params = yaml.load(f)

        data = ef.ExptFitter.package_data(ld, ld.metadata, exp_params)

        xvals, y, cal = ef.Calibrate1Q_cal(data, ld.metadata, ch)

        # Calculate the averages of the RB sequences
        sequence_ids = set([xval[0] for xval in xvals])
        individual_ids = set([xval[1] for xval in xvals])

        rb_avg = []

        for ii in individual_ids:
            sum_ = 0.0
            cnt_ = 0
            for jj in sequence_ids:
                sum_ += y[[a.all() for a in xvals == [float(jj), float(ii)]]][0]
                cnt_ += 1

            rb_avg.append(sum_/cnt_)

        return rb_avg


    def single_length_RB(self,Rb_number, n_max, n=1):
        """
        Use the qsoft backend to generate a RB sequence as a qasm string.

        :param n_max: number of maximum cliffords
        :param n: number of qubits on which to do simultaneous RB
        """
        sequences = []
        for i in range(Rb_number):
            cmd = '~/sw-backend-qsoft-legacy/benchmarking-qasm1.1/rb_stream '
            cmd += '~/sw-backend-qsoft-legacy/setup/benchmarking/x90y90_basis.qasm [{}] {} {} 1'.format(n, n_max, n_max)
            qasm = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            sequences.append(qasm)
        self.rb_length = n_max

        return sequences

    def make_template_sqore_file(self, sequences, parameter_batch, sqore_dir):
        """
        For a single set of pulse parameters we run rep different RB
        sequences of some length.
        :param sequences: List of sqore objects containing RB sequences
        :param parameter_batch: list of dictionairies of pulse parameters and values.
        :param sqore_dir: the directory where the sqore file will be saved.
        """
        # prepare the calibration string

        final_string = ''

        # insert parameter solutions:

        include_cals = True
        for ii in range(len(sequences)):
            qasm = sequences[ii]
            sqore = build_sqore(qasm, 0)
            sqore.print_sq2()

            # extract string from sqore obj to change it
            sqore_string = StringIO()
            sqore.print_sq2(sqore_string)
            sqore_lines = sqore_string.getvalue().split('\n')

            if include_cals:
                cal_sqore = build_sqore(qasm, 3)

                # extract string from sqore obj to change it
                cal_sqore_string = StringIO()
                cal_sqore.print_sq2(cal_sqore_string)
                cal_sqore_lines = cal_sqore_string.getvalue().split('\n')

            new_sqore_string = ''

            for jj in range(len(parameter_batch)):

                if include_cals:
                    lines = cal_sqore_lines
                    include_cals = False
                else:
                    lines = sqore_lines

                param_dict = parameter_batch[jj]
                new_sqore_lines = []

                for line in lines:
                    seq_elements = line.split(';')
                    ch = seq_elements[0]

                    new_seq = [ch]

                    for gate in seq_elements[1:]:
                        gate = gate.replace(' ', '')

                        parameters = get_parameters(ch, gate, param_dict)

                        new_seq.append(gate + parameters)

                    new_line = '; '.join(new_seq)

                    new_line = new_line.replace('xval(0,{})'.format(self.rb_length), 'xval({},{})'.format(ii, jj))

                    new_sqore_lines.append(new_line)
                new_sqore_string += '\n'.join(new_sqore_lines)
            final_string += new_sqore_string + '\n'

        sq2file = open(sqore_dir + "CMASearch.sq2", "w")
        sq2file.write(final_string)
        sq2file.close()

