# test_SYCL
SYCL testing.
# Compiling and running
## Compiling
```
$ source /path_to/inteloneapi/setvars.sh
$ mkdir build && cd build
$ cmake /path_to/hello_sycl
$ make
```
## Running
By default, there is no kernel execution.
At least one of `-b n`, `-c n` or `-g n` should be used to run `n` threads
with CPU+GPU, CPU-only or GPU-only kernels, respectively, executing in each
of the `n` threads.
By default, each of the `n` threads, specified above, executes 2 concurrent
events/kernels.
The number of kernels to execute can be set using `-B k`, `-C k` or `-G k`,
as per the `-b n`, `-c n` and `-g n` threads, respectively.
### Example 1
Run 1 thread that executes 4 CPU and 4 GPU kernels:
```
./tbb-task-sycl -b 1 -B 4
```
### Example 2
Run 2 threads that executes 2 CPU and 2 GPU kernels per thread:
```
./tbb-task-sycl -b 2 [-B 2]
```
Note that `-B 2` is optional as 2 kernels is the default.
### Example 3
Run 4 threads that executes 4 CPU kernels in each thread:
```
./tbb-task-sycl -c 4 -C 4
```
### Example 4
Run 2 threads that executes 4 GPU kernels in each thread:
```
./tbb-task-sycl -g 2 -G 4
```
### Example 5
Run 2 CPU threads with 4 CPU kernels, and 2 GPU threads with 2 GPU kernels:
```
./tbb-task-sycl -c 2 -C 4 -g 2 -G 2
```

# Test results
All results shown here were performed on a MacBook13,3 using the following:
- Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz
- Intel(R) Gen9 HD Graphics NEO
## Test 1
Varying the number of CPU+GPU/CPU/GPU threads and their respective number
of kernels yields very different hardware utilization, even when the total
number of events, n_threads x n_kernels, is the same.
For example,
```
$ ./tbb-task-sycl -c 1 -C 4
...
sycl_task::CPU0 time [ms]: 35065
========= ALL DONE =========
otal execution time [ms]: 35065
```
and
```
$ ./tbb-task-sycl -c 4 -C 1
...
sycl_task::CPU3 time [ms]: 13038
sycl_task::CPU2 time [ms]: 25984
sycl_task::CPU1 time [ms]: 48487
sycl_task::CPU0 time [ms]: 48514
========= ALL DONE =========
Total execution time [ms]: 48514
```
These two runs above process the same number of events (4) but in
the first -- where a single thread executes 4 kernels -- all physical
cores max out on my system, while in the second -- where 4 thread each execute
one kernel -- only a single core seems to be used at a given time.
## Test 2
Similarly,
```
$ ./tbb-task-sycl -b 1 -B 4
...
sycl_task::CPU+GPU0 time [ms]: 39923
========= ALL DONE =========
otal execution time [ms]: 39924
```
and
```
$ ./tbb-task-sycl -c 2 -C 2 -g 2 -G 2
...
sycl_task::GPU1 time [ms]: 8288
sycl_task::CPU1 time [ms]: 33350
sycl_task::GPU0 time [ms]: 40811
sycl_task::CPU0 time [ms]: 56533
========= ALL DONE =========
otal execution time [ms]: 56533
```
each execute 4 CPU and 4 GPU kernels, but using multiple threads to execute each
set of kernels has a degrading effect as opposed to running everything in a
single thread.
