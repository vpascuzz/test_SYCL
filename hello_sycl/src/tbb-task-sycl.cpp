// Local include(s)

// SYCL include(s)
#include <CL/sycl.hpp>

// TBB include(s)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

// System include(s)
#include <atomic>
#include <chrono>
#include <memory>
#include <iostream>
#include <unistd.h>

// Less typing
namespace sycl = cl::sycl;

// Forward declaration for command group handler ::parallel_for template.
// Could use -fsycl-unnamed-lambda but doesn't work with ComputeCpp.
class DummyGPU;
class DummyCPU;
class DummyGPU_singlet;
class DummyCPU_singlet;
const std::size_t BUFFER_SIZE = 1024;

// sycl_task
class sycl_task
{
public:
    // Constructor
    sycl_task(std::string name, std::size_t numEvents = 2)
        : m_name(name), m_numEvents(numEvents)
    {
    }

    // Operation/task definition
    void operator()() const
    {
        // Start the timer.
        auto start = std::chrono::high_resolution_clock::now();

        { //starting SYCL block
            // Create a buffer `BUFFER_SIZE` floating point numbers.
            cl::sycl::range<1> n_items{BUFFER_SIZE};
            sycl::buffer<sycl::cl_float> buffer(n_items);

            // Create a queue from the selector.
            std::unique_ptr<sycl::queue> q;
            std::unique_ptr<sycl::queue> q_cpu;
            std::unique_ptr<sycl::queue> q_gpu;

            // Create a `device_selector` depending on `m_name`
            std::unique_ptr<sycl::device_selector> device_selector;
            std::unique_ptr<sycl::cpu_selector> cpu_selector;
            std::unique_ptr<sycl::gpu_selector> gpu_selector;
            if (m_name.find("CPU+GPU") != std::string::npos)
            {
                cpu_selector.reset(new sycl::cpu_selector);
                gpu_selector.reset(new sycl::gpu_selector);
                q_cpu.reset(new sycl::queue(*cpu_selector));
                q_gpu.reset(new sycl::queue(*gpu_selector));
            }
            else if (m_name.find("GPU") != std::string::npos)
            {
                device_selector.reset(new sycl::gpu_selector);
                q.reset(new sycl::queue(*device_selector));
            }
            else if (m_name.find("CPU") != std::string::npos)
            {
                device_selector.reset(new sycl::cpu_selector);
                q.reset(new sycl::queue(*device_selector));
            }

            // Run a calculation on the buffer.
            if (q_cpu && q_gpu)
            {
                // Logging
                std::cout << "sycl_task::" << m_name << " executing using:" << std::endl;
                std::cout << " -- " << q_cpu->get_device().get_info<sycl::info::device::name>() << std::endl;
                std::cout << " -- " << q_gpu->get_device().get_info<sycl::info::device::name>() << std::endl;

                // Loop
                for (std::size_t i = 0; i < m_numEvents; ++i)
                {
                    auto event_cpu = q_cpu->submit([&](sycl::handler &cgh) {
                        auto acc = buffer.get_access<sycl::access::mode::read>(cgh);
                        cgh.parallel_for<class DummyCPU>(n_items, [=](sycl::id<1> idx) {
                            // The ranges of these loops are set to give reasonably length processing times.
                            for (int index = 0; index < 1000; ++index)
                            {
                                volatile unsigned long long i;
                                for (i = 0; i < 1000000ULL; ++i)
                                    ;
                            } // end long loop
                        });   // end of kernel/parallel_for
                    });       // end of sycl queue commands
                    auto event_gpu = q_gpu->submit([&](sycl::handler &cgh) {
                        auto acc = buffer.get_access<sycl::access::mode::read>(cgh);
                        cgh.parallel_for<class DummyGPU>(n_items, [=](sycl::id<1> idx) {
                            // The ranges of these loops are set to give reasonably length processing times.
                            for (int index = 0; index < 100; ++index)
                            {
                                volatile unsigned long long i;
                                for (i = 0; i < 100000ULL; ++i)
                                    ;
                            } // end long loop
                        });   // end of kernel/parallel_for
                    });       // end of sycl queue commands
                }             // end group loop
            }                 // end GPU processing
            else if (q->get_device().is_gpu() || q->get_device().is_accelerator())
            {
                // Logging
                std::cout << "sycl_task::" << m_name << " executing using:" << std::endl;
                std::cout << " -- " << q->get_device().get_info<sycl::info::device::name>() << std::endl;

                // Loop
                for (std::size_t i = 0; i < m_numEvents; ++i)
                {
                    auto event = q->submit([&](sycl::handler &cgh) {
                        auto acc = buffer.get_access<sycl::access::mode::read>(cgh);
                        cgh.parallel_for<class DummyGPU_singlet>(n_items, [=](sycl::id<1> idx) {
                            // The ranges of these loops are set to give reasonably length processing times.
                            for (int index = 0; index < 100; ++index)
                            {
                                volatile unsigned long long i;
                                for (i = 0; i < 100000ULL; ++i)
                                    ;
                            } // end long loop
                        });   // end of kernel/parallel_for
                    });       // end of sycl queue commands
                }             // end group loop
            }                 // end GPU processing
            else if (q->get_device().is_cpu())
            {
                // Logging
                std::cout << "sycl_task::" << m_name << " executing using:" << std::endl;
                std::cout << " -- " << q->get_device().get_info<sycl::info::device::name>() << std::endl;

                // Loop
                for (std::size_t i = 0; i < m_numEvents; ++i)
                {
                    auto event = q->submit([&](sycl::handler &cgh) {
                        auto acc = buffer.get_access<sycl::access::mode::read>(cgh);
                        cgh.parallel_for<class DummyCPU_singlet>(n_items, [=](sycl::id<1> idx) {
                            // The ranges of these loops are set to give reasonably length processing times.
                            for (int index = 0; index < 1000; ++index)
                            {
                                volatile unsigned long long i;
                                for (i = 0; i < 1000000ULL; ++i)
                                    ;
                            } // end long loop
                        });   // end of kernel/parallel_for
                    });       // end of sycl queue commands
                }             // end group loop
            }                 // end CPU processing
        }                     // end of sycl scope; wait until queued work completes

        // Stop the timer and print the result.
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "sycl_task::" << m_name << " time [ms]: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000
                  << std::endl;
    } // operator()

private:
    // Name of this task.
    const std::string m_name;
    // Number of events/kernels to execute.
    const std::size_t m_numEvents;
}; // gpu_task

// tbb_task
class tbb_task
{
public:
    /// Constructor
    tbb_task(std::string name)
        : m_name(name)
    {
    }
    /// Operation/task definition
    void operator()() const
    {
        // Print a welcome message.
        std::cout << "tbb_task::" << m_name << " executing..." << std::endl;

        // Start the timer.
        auto start = std::chrono::high_resolution_clock::now();

        std::array<float, BUFFER_SIZE> buffer;
        tbb::parallel_for(tbb::blocked_range<int>(0, buffer.size()),
                          [&](tbb::blocked_range<int> r) {
                              // The ranges of these loops are set to give reasonably length processing times.
                              for (int index = 0; index < 1000; ++index)
                              {
                                  volatile unsigned long long i;
                                  for (i = 0; i < 10000ULL; ++i)
                                      ;
                              } // end long loop
                          });   // end of kernel/parallel_for

        // Stop the timer and print the result.
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "tbb_task::" << m_name << " time [us]: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
                  << std::endl;
    } // operator()

private:
    /// Name of this task.
    const std::string m_name;
}; // tbb_task

// Main
int main(int argc, char **argv)
{
    int opt(0);
    // Number of tasks/threads to run
    std::size_t num_tasks(0);
    // Number of tasks/threads to run both cpu and gpu
    std::size_t num_tasks_cpugpu(0);
    // Number of tasks/threads to run only cpu
    std::size_t num_tasks_cpu(0);
    // Number of tasks/threads to run only gpu
    std::size_t num_tasks_gpu(0);
    // Number of events per cpu+gpu thread
    std::size_t num_events_cpugpu(2);
    // Number of events per cpu thread
    std::size_t num_events_cpu(2);
    // Number of events per gpu thread
    std::size_t num_events_gpu(2);
    // Display help
    bool help(false);

    while ((opt = getopt(argc, argv, "hb:c:g:B:C:G:")) != -1)
    {
        switch (opt)
        {
        case 'b':
            num_tasks_cpugpu = atoi(optarg);
            break;
        case 'c':
            num_tasks_cpu = atoi(optarg);
            break;
        case 'g':
            num_tasks_gpu = atoi(optarg);
            break;
        case 'B':
            num_events_cpugpu = atoi(optarg);
            break;
        case 'C':
            num_events_cpu = atoi(optarg);
            break;
        case 'G':
            num_events_gpu = atoi(optarg);
            break;
        case 'h':
            help = true;
        default: /* '?' */
            fprintf(stderr, "Usage: %s [-h] [-bcg N] [-BCG k]\n", argv[0]);
            if (help)
            {
                printf("         -b n: run 'n' threads consisting of 'b'oth CPU and GPU kernels in each thread\n");
                printf("         -c n: run 'n' threads consisting of 'c'pu-only kernels in each thread\n");
                printf("         -g n: run 'n' threads consisting of 'g'pu-only kernels in each thread\n");
                printf("         -B k: run 'k' kernels in 'B'oth cpu and gpu mode (-b)\n");
                printf("         -C k: run 'k' kernels in 'C'pu-only mode (-c)\n");
                printf("         -G k: run 'k' kernels in 'G'pu-only mode (-g)\n\n");
            }

            exit(EXIT_FAILURE);
        }
    }

    // Define tbb::task_group for running CPU and GPU tasks
    tbb::task_group tg;

    // Start the timer.
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the tasks.
    for (std::size_t i = 0; i < num_tasks_cpugpu; ++i) // CPU+GPU tasks
    {
        const std::string name = "CPU+GPU" + std::to_string(i);
        tg.run(sycl_task(name, num_events_cpugpu)); // spawn task and return
    }
    for (std::size_t i = 0; i < num_tasks_cpu; ++i) // CPU-only tasks
    {
        const std::string name = "CPU" + std::to_string(i);
        tg.run(sycl_task(name, num_events_cpu)); // spawn task and return
    }
    for (std::size_t i = 0; i < num_tasks_gpu; ++i) // GPU-only tasks
    {
        const std::string name = "GPU" + std::to_string(i);
        tg.run(sycl_task(name, num_events_gpu)); // spawn task and return
    }
    tg.wait(); // wait for tasks to complete

    std::cout << "========= ALL DONE =========" << std::endl;

    // Stop the timer and print the result.
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Total execution time [ms]: "
              << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000
              << std::endl;

    // Graceful exit.
    return 0;
}
