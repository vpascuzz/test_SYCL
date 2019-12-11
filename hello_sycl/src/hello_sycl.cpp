


// Local include(s)


// SYCL include(s)
#include <CL/sycl.hpp>

// System include(s)
#include <iostream>
#include <chrono>

namespace sycl = cl::sycl;

// Forward declaration for command group handler ::parallel_for
class vector_addition;

int main(int, char**) {
  sycl::float8 a = { 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0 };
  sycl::float8 b = { 4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0 };
  sycl::float8 c = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  sycl::default_selector device_selector;

  sycl::queue queue(device_selector);
  std::cout << "Running on "
    << queue.get_device().get_info<sycl::info::device::name>()
    << "\n";
  {
    sycl::buffer<cl::sycl::float8, 1> a_sycl(&a, sycl::range<1>(1));
    sycl::buffer<cl::sycl::float8, 1> b_sycl(&b, sycl::range<1>(1));
    sycl::buffer<cl::sycl::float8, 1> c_sycl(&c, sycl::range<1>(1));
    //
    //      /**
    //       * single-task submission
    //       */
    //      auto start = std::chrono::high_resolution_clock::now();
    //      queue.submit([&] (cl::sycl::handler& cgh) {
    //         auto a_acc = a_sycl.get_access<cl::sycl::access::mode::read>(cgh);
    //         auto b_acc = b_sycl.get_access<cl::sycl::access::mode::read>(cgh);
    //         auto c_acc = c_sycl.get_access<cl::sycl::access::mode::discard_write>(cgh);
    //
    //         cgh.single_task<class vector_addition>([=] () {
    //         c_acc[0] = a_acc[0] + b_acc[0];
    //         });
    //      });
    //      auto stop = std::chrono::high_resolution_clock::now();
    //      std::cout << "vector_addition time: "
    //    		  	  << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() / 1000
    //				  << std::endl;

    /**
     * parallel_for submission
     */
    auto start = std::chrono::high_resolution_clock::now();
    queue.submit([&] (sycl::handler& cgh) {
        auto a_acc = a_sycl.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b_sycl.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c_sycl.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<class vector_addition>(
            cl::sycl::range<1>{16}, [=] (sycl::id<1> id) {
            c_acc[id] = a_acc[id] + b_acc[id];
//            printf("id: %lu\n", id.get(0));
            });
        });

    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "vector_addition time [ms]: "
      << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() / 1000
      << std::endl;
  }

  std::cout	<< "  A { "
    << a.s0() << ", "
    << a.s1() << ", "
    << a.s2() << ", "
    << a.s3() << ", "
    << a.s4() << ", "
    << a.s5() << ", "
    << a.s6() << ", "
    << a.s7() << " }\n"
    << "  B { "
    << b.s0() << ", "
    << b.s1() << ", "
    << b.s2() << ", "
    << b.s3() << ", "
    << b.s4() << ", "
    << b.s5() << ", "
    << b.s6() << ", "
    << b.s7() << " }\n"
    << "------------------\n"
    << "  C { "
    << c.s0() << ", "
    << c.s1() << ", "
    << c.s2() << ", "
    << c.s3() << ", "
    << c.s4() << ", "
    << c.s5() << ", "
    << c.s6() << ", "
    << c.s7() << " }\n"
    << std::endl;

  return 0;
}
