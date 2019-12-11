# test_SYCL
SYCL testing.
# Compile and run
## Compile
```
$ source /path/to/inteloneapi/setvars.sh
$ cmake /path/to/CMakeLists.txt
$ make
```
## Run
```
./tbb-task-sycl
```
# Notes
No CLI arguments are used; must change task settings by-hand, e.g. in `hello_sycl/src/tbb-task-sycl.cpp`:
```
const std::size_t NUM_TASKS_CPUGPU = 0;
const std::size_t NUM_TASKS_GPU = 1;
const std::size_t NUM_TASKS_CPU = 1;
const std::size_t NUM_EVENTS_CPUGPU = 2;
const std::size_t NUM_EVENTS_GPU = 2;
const std::size_t NUM_EVENTS_CPU = 2;
```
Sorry!