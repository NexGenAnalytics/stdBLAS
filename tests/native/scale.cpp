#include <experimental/linalg>
#include <experimental/mdspan>

// FIXME I can't actually test the executor overloads, since my GCC
// (9.1.0, via Homebrew) isn't set up correctly:
//
// .../gcc/9.1.0/include/c++/9.1.0/pstl/parallel_backend_tbb.h:19:10: fatal error: tbb/blocked_range.h: No such file or directory
//   19 | #include <tbb/blocked_range.h>
//      |          ^~~~~~~~~~~~~~~~~~~~~

//#include <execution>
#include <vector>
#include "gtest/gtest.h"

namespace {
  using std::experimental::dynamic_extent;
  using std::experimental::extents;
  using std::experimental::mdspan;
  using std::experimental::linalg::scale;

  TEST(BLAS1_scale, mdspan_double)
  {
    using scalar_t = double;
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);

    {
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k = scalar_t (k) + 1.0;
        x(k) = x_k;
      }
      const scalar_t scaleFactor = 5.0;
      scale(scaleFactor, x);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k = scalar_t (k) + 1.0;
        EXPECT_EQ( x(k), scaleFactor * x_k );
      }
    }
    {
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k = scalar_t (k) + 1.0;
        x(k) = x_k;
      }
      const float scaleFactor = 5.0;
      scale(scaleFactor, x);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k = scalar_t (k) + 1.0;
        EXPECT_EQ( x(k), scaleFactor * x_k );
      }
    }
  }

  TEST(BLAS1_scale, mdspan_complex_double)
  {
    using real_t = double;
    using scalar_t = std::complex<real_t>;
    using vector_t = mdspan<scalar_t, extents<dynamic_extent>>;

    constexpr std::size_t vectorSize(5);
    constexpr std::size_t storageSize = vectorSize;
    std::vector<scalar_t> storage(storageSize);

    vector_t x(storage.data(), vectorSize);

    {
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        x(k) = x_k;
      }
      const real_t scaleFactor = 5.0;
      scale(scaleFactor, x);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        EXPECT_EQ( x(k), scaleFactor * x_k );
      }
    }
    {
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        x(k) = x_k;
      }
      const scalar_t scaleFactor (5.0, -1.0);
      scale(scaleFactor, x);
      for (std::size_t k = 0; k < vectorSize; ++k) {
        const scalar_t x_k(real_t(k) + 4.0, -real_t(k) - 1.0);
        EXPECT_EQ( x(k), scaleFactor * x_k );
      }
    }
  }
}

// int main() {
//   std::cout << "hello world" << std::endl;
// }
