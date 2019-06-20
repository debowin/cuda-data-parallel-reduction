# cuda-data-parallel-reduction

## Overview

This is an implementation of a work-efficient parallel reduction algorithm on the GPU with accumulation using
atomic additions. An alternative version would invoke the reduction kernal more than once in a hierarchical manner so as to
further reduce the block sums computed by the previous kernel invocations.

## Execution

* Run "make" to build the executable of this file.
* For debugging, run "make dbg=1" to build a debuggable version of the executable binary.
* Run the binary using "./~name-of-the-artifact~"

There are several modes of operation for the application -

* *No arguments*: The application will create a randomly-initialized array
to process. After the device kernel is invoked, it will compute the correct
solution value using the CPU and compare it with the device-computed
solution. If the solutions match (within a certain tolerance), it will print
out "Test PASSED" to the screen before exiting.

* *One argument*: The application will initialize the input array with the
values found in the file specified by the argument.

In either mode, the program will print out the final result of the CPU and GPU
computations, and whether or not the comparison passed.
