import argparse
import numpy as np
import configparser
from colorama import Fore, Back, Style 

from test_bb import test_bb
from test_cell_ids import test_cell_ids
from test_grid_ds import test_grid_ds
from test_find_neighs import test_find_neighs
from test_pd_pooling import test_pd_pooling
from test_pdf import test_pdf
from test_pdf_grads import test_pdf_grads
from test_mcconv import test_mcconv
from test_mcconv_grads import test_mcconv_grads
from test_mcconv_pt_grads import test_mcconv_pt_grads
from test_kpconv import test_kpconv
from test_kpconv_grads import test_kpconv_grads
from test_kpconv_pt_grads import test_kpconv_pt_grads

test_dictionary = {
    'bounding_box': test_bb,
    'cell_ids': test_cell_ids,
    'build_grid_ds': test_grid_ds,
    'find_neighs': test_find_neighs,
    'pd_pooling': test_pd_pooling,
    'pdf': test_pdf,
    'pdf_grads': test_pdf_grads,
    'mcconv': test_mcconv,
    'mcconv_grads': test_mcconv_grads,
    'mcconv_pt_grads': test_mcconv_pt_grads,
    'kpconv': test_kpconv,
    'kpconv_grads': test_kpconv_grads,
    'kpconv_pt_grads': test_kpconv_pt_grads
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unit tests')
    parser.add_argument('--conf_file', default='unit_test.ini', help='Config file (default: unit_test.ini)')
    args = parser.parse_args()

    # Load the test data.
    dataset = np.full((32, 100000, 3), 0.0, dtype = np.float32)
    for modelI in range(32):
        cur_model = np.loadtxt("point_clouds/model"+str(modelI)+".xyz")
        max_pt = np.amax(cur_model, axis=0)
        min_pt = np.amin(cur_model, axis=0)
        center = (max_pt + min_pt)*0.5
        diagonal = max_pt - min_pt
        aabb_size = np.linalg.norm(diagonal)
        dataset[modelI, :, :] = (cur_model - center)/aabb_size

    # Load the config file.
    config_parser = configparser.ConfigParser()
    config_parser.read(args.conf_file)
    num_tests = int(config_parser['UnitTests']['num_tests'])

    # Evaluate tests.
    print()
    print("########### Num tests", num_tests)
    print()
    passed = 0
    for cur_test in range(num_tests):

        # Config dictionary of the test.
        cur_config_dict = config_parser['Test_'+str(cur_test+1)]

        # Get test function.
        test_type = cur_config_dict['type']
        test_funct = test_dictionary[test_type]

        # Evaluate the test.
        result, time = test_funct(cur_config_dict, dataset)

        if result:
            passed += 1
            print(str(cur_test+1)+"/"+str(num_tests)+" ["+test_type+"] ("+\
                "{:.2f} ms".format(time)+")"+Fore.GREEN+" Success"+Style.RESET_ALL)
        else:
            print(str(cur_test+1)+"/"+str(num_tests)+" ["+test_type+"] ("+\
                "{:.2f} ms".format(time)+")"+Fore.RED+" Error"+Style.RESET_ALL)
    
    # Print final result.
    print()
    print("Total: "+str(passed)+" / "+str(num_tests))
    print()