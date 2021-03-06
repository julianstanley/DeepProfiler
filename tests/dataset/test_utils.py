import deepprofiler.dataset.utils
import pytest
import sys
import os
from io import StringIO
import multiprocessing
import shutil

def test_printProgress():
    # setup the environment
    backup = sys.stdout
    test_iterations = [-1,0,1,2,3,4,5]
    test_outputs = []
    expected_outputs = ['\rError: printProgress() function received a negative "iteration" value.',
                        '\rProgress |' + '-' * 100 +'| 0.0% Complete',
                        '\rProgress |' + '#' * 25 + '-' * 75 +'| 25.0% Complete',
                        '\rProgress |' + '#' * 50 + '-' * 50 +'| 50.0% Complete',
                        '\rProgress |' + '#' * 75 + '-' * 25 +'| 75.0% Complete',
                        '\rProgress |' + '#' * 100 +'| 100.0% Complete\n',
                        '\rError: printProgress() function received an "iteration" value greater than the "total" value.']
    test_total = 4
    for iteration in test_iterations:

        # ####
        sys.stdout = StringIO()     # capture output
        deepprofiler.dataset.utils.printProgress(iteration,test_total)
        out = sys.stdout.getvalue() # release output
        #####
        
        test_outputs.append(out)
        
        assert test_outputs[iteration + 1] == expected_outputs[iteration + 1]
    
    test_iteration = 1
    test_total = -1
    expected_output = '\rError: printProgress() function received a negative "total" value.'
    

    # ####
    sys.stdout = StringIO()     # capture output
    deepprofiler.dataset.utils.printProgress(test_iteration,test_total)
    out = sys.stdout.getvalue() # release output
    #####
                
    assert out == expected_output 
    
    test_total = 2
    test_barLength = -100
    expected_output = '\rError: printProgress() function received a negative "barLength" value.'
    

    # ####
    sys.stdout = StringIO()     # capture output
    deepprofiler.dataset.utils.printProgress(test_iteration,test_total,barLength=test_barLength)
    out = sys.stdout.getvalue() # release output
    #####
                
    assert out == expected_output

    test_total = -2
    test_iteration = -1
    test_barLength = -100
    expected_output = '\rError: printProgress() function received multiple negative values.'
    

    # ####
    sys.stdout = StringIO()     # capture output
    deepprofiler.dataset.utils.printProgress(test_iteration,test_total,barLength=test_barLength)
    out = sys.stdout.getvalue() # release output
    #####
                
    assert out == expected_output
    
    test_total = 2
    test_iteration = 1
    test_barLength = 200
    test_decimals = 2
    test_prefix = 'Dog'
    test_suffix = 'Cat'
    expected_output = '\rDog |' + '#' * 100 + '-' * 100 +'| 50.00% Cat'
    

    # ####
    sys.stdout = StringIO()     # capture output
    deepprofiler.dataset.utils.printProgress(test_iteration,test_total,prefix=test_prefix,suffix=test_suffix,decimals=test_decimals,barLength=test_barLength)
    out = sys.stdout.getvalue() # release output
    #####
                
    assert out == expected_output
    
    sys.stdout.close()  # close the stream 
    sys.stdout = backup # restore original stdout

def test_check_path(): 
    #feeds check_path() a made up filename and checks if the function creates the directory for that file

    test_filename = "tmp/filefolder/dog/cat/test.jpg"
    deepprofiler.dataset.utils.check_path(test_filename)
    assert os.path.isdir("tmp/filefolder/dog/cat/") 
    shutil.rmtree("tmp")   

#tests for class Parallel

def test_init(): 
    cpus =  multiprocessing.cpu_count()
    parallel = deepprofiler.dataset.utils.Parallel([])
    assert parallel.fixed_args == []
    assert parallel.pool._processes == cpus
    parallel = deepprofiler.dataset.utils.Parallel([1,2,3],numProcs=0)
    assert parallel.fixed_args == [1,2,3]
    assert parallel.pool._processes == cpus
    parallel = deepprofiler.dataset.utils.Parallel([1,2,3],numProcs=cpus+1)
    assert parallel.fixed_args == [1,2,3]
    assert parallel.pool._processes == cpus
    if cpus > 1:
        parallel = deepprofiler.dataset.utils.Parallel([1,2,3],numProcs=cpus-1)
        assert parallel.fixed_args == [1,2,3]
        assert parallel.pool._processes == cpus-1
    
#def test_compute():
    #can't test because can't serialize function
    
#def test_tic():
    #pass

#def test_toc():
    #pass
    
#class Logger() does not requires tests
