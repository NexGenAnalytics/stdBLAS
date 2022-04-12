 /*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_KOKKOSKERNELS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_

#include <complex>
#include "signal_kokkos_impl_called.hpp"

namespace KokkosKernelsSTD {

namespace Impl {

// manages parallel execution of independent action
// called like action(i, j) for each matrix element A(i, j)
template <typename ExecSpace, typename MatrixType>
class ParallelMatrixVisitor {
public:
  KOKKOS_INLINE_FUNCTION ParallelMatrixVisitor(ExecSpace &&exec_in, MatrixType &A_in):
    exec(exec_in), A(A_in), ext0(A.extent(0)), ext1(A.extent(1))
  {}

  template <typename ActionType>
  void for_each_triangle_matrix_element(std::experimental::linalg::upper_triangle_t t, ActionType action) {
    Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, ext1),
      KOKKOS_LAMBDA(const auto j) {
        using idx_type = std::remove_const_t<decltype(j)>;
        for (idx_type i = 0; i <= j; ++i) {
          action(i, j);
        }
      });
    exec.fence();
  }

  template <typename ActionType>
  void for_each_triangle_matrix_element(std::experimental::linalg::lower_triangle_t t, ActionType action) {
    for_each_triangle_matrix_element(std::experimental::linalg::upper_triangle,
        [action](const auto i, const auto j) {
          action(j, i);
      });
  }

private:

  template <typename T>
  KOKKOS_INLINE_FUNCTION
  static bool is_int_mul_safe(T a, T b) {
    static constexpr auto max_val = std::numeric_limits<decltype(a * b)>::max();
    return a <= max_val / b;
  }

private:
  const ExecSpace &exec;
  MatrixType &A;
  const size_t ext0;
  const size_t ext1;
};

} // namespace Impl

namespace Impl {

// Note: phrase it simply and the same as in specification ("has unique layout")
template <typename Layout,
          std::experimental::extents<>::size_type numRows,
          std::experimental::extents<>::size_type numCols>
inline constexpr bool is_unique_layout_v = Layout::template mapping<
    std::experimental::extents<numRows, numCols> >::is_always_unique();

template <typename Layout>
struct is_layout_blas_packed: public std::false_type {};

template <typename Triangle, typename StorageOrder>
struct is_layout_blas_packed<
  std::experimental::linalg::layout_blas_packed<Triangle, StorageOrder>>:
    public std::true_type {};

template <typename Layout>
inline constexpr bool is_layout_blas_packed_v = is_layout_blas_packed<Layout>::value;

// Note: will only signal failure for layout_blas_packed with diferent triangle
template <typename Layout, typename Triangle>
struct triangle_layout_match: public std::true_type {};

template <typename StorageOrder, typename Triangle1, typename Triangle2>
struct triangle_layout_match<
  std::experimental::linalg::layout_blas_packed<Triangle1, StorageOrder>,
  Triangle2>
{
  static constexpr bool value = std::is_same_v<Triangle1, Triangle2>;
};

template <typename Layout, typename Triangle>
inline constexpr bool triangle_layout_match_v = triangle_layout_match<Layout, Triangle>::value;

};

// Rank-1 update of a Symmetric matrix
// performs BLAS xSYR/xSPR: A[i,j] += x[i] * x[j]

template<class ExecSpace,
         class ElementType_x,
         std::experimental::extents<>::size_type ext_x,
         class Layout_x,
         class ElementType_A,
         std::experimental::extents<>::size_type numRows_A,
         std::experimental::extents<>::size_type numCols_A,
         class Layout_A,
         class Triangle>
  requires (Impl::is_unique_layout_v<Layout_A, numRows_A, numCols_A>
            or Impl::is_layout_blas_packed_v<Layout_A>)
void symmetric_matrix_rank_1_update(kokkos_exec<ExecSpace> &&exec,
  std::experimental::mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x,
    std::experimental::default_accessor<ElementType_x>> x,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A,
    std::experimental::default_accessor<ElementType_A>> A,
  Triangle t)
{
  // constraints
  static_assert(A.rank() == 2);
  static_assert(x.rank() == 1);
  static_assert(Impl::triangle_layout_match_v<Layout_A, Triangle>);

  // preconditions
  if ( A.extent(0) != A.extent(1) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_1_update: A.extent(0) != A.extent(1)");
  }
  if ( A.extent(0) != x.extent(0) ){
    throw std::runtime_error("KokkosBlas: symmetric_matrix_rank_1_update: A.extent(0) != x.extent(0)");
  }

  Impl::signal_kokkos_impl_called("symmetric_matrix_rank1_update");

  auto x_view = Impl::mdspan_to_view(x);
  auto A_view = Impl::mdspan_to_view(A);
  Impl::ParallelMatrixVisitor v(ExecSpace(), A_view);
  v.for_each_triangle_matrix_element(t,
    KOKKOS_LAMBDA(const auto i, const auto j) {
      A_view(i, j) += x_view(i) * x_view(j);
    });
}

} // namespace KokkosKernelsSTD
#endif
