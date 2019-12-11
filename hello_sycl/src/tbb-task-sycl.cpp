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
        // Logging
        std::cout << "sycl_task::" << m_name << " executing..." << std::endl;

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
int main(int, char **)
{
    // Number of tasks to run in tbb::task_group
    const std::size_t NUM_TASKS_CPUGPU = 0;
    const std::size_t NUM_TASKS_GPU = 1;
    const std::size_t NUM_TASKS_CPU = 1;
    const std::size_t NUM_EVENTS_CPUGPU = 2;
    const std::size_t NUM_EVENTS_GPU = 2;
    const std::size_t NUM_EVENTS_CPU = 2;

    // Define tbb::task_group for running CPU and GPU tasks
    tbb::task_group tg;

    // Execute the tasks.
    for (std::size_t i = 0; i < NUM_TASKS_CPUGPU; ++i)  // CPU+GPU tasks
    {
        const std::string name = "CPU+GPU" + std::to_string(i);
        tg.run(sycl_task(name, NUM_EVENTS_CPUGPU)); // spawn task and return
    }
    for (std::size_t i = 0; i < NUM_TASKS_CPU; ++i)     // CPU-only tasks
    {
        const std::string name = "CPU" + std::to_string(i);
        tg.run(sycl_task(name, NUM_EVENTS_CPU)); // spawn task and return
    }
    for (std::size_t i = 0; i < NUM_TASKS_GPU; ++i)     // GPU-only tasks
    {
        const std::string name = "GPU" + std::to_string(i);
        tg.run(sycl_task(name, NUM_EVENTS_GPU)); // spawn task and return
    }
    tg.wait(); // wait for tasks to complete

    std::cout << "========= ALL DONE =========" << std::endl;

    // Graceful exit.
    return 0;
}
