/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_

namespace std {
namespace experimental {
inline namespace __p1673_version_0 {
namespace linalg {

// Nonsymmetric non-conjugated rank-1 update

template<class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update(
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  using size_type = typename extents<>::size_type;
  for (size_type i = 0; i < A.extent(0); ++i) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      A(i,j) += x(i) * y(j);
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  matrix_rank_1_update(x, y, A);
}

// Nonsymmetric conjugated rank-1 update

template<class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update_c(
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  matrix_rank_1_update(x, conjugated(y), A);
}

template<class ExecutionPolicy,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_y,
         extents<>::size_type ext_y,
         class Layout_y,
         class Accessor_y,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A>
void matrix_rank_1_update_c(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_y, std::experimental::extents<ext_y>, Layout_y, Accessor_y> y,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A)
{
  matrix_rank_1_update_c(x, y, A);
}

// Rank-1 update of a Symmetric matrix

template<class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle>
void symmetric_matrix_rank_1_update(
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using size_type = typename extents<>::size_type;
  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = j; i < A.extent(0); ++i) {
        A(i,j) += x(i) * x(j);
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i <= j; ++i) {
        A(i,j) += x(i) * x(j);
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle>
void symmetric_matrix_rank_1_update(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  symmetric_matrix_rank_1_update(x, A, t);
}

// Rank-1 update of a Hermitian matrix

template<class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle>
void hermitian_matrix_rank_1_update(
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle /* t */)
{
  using std::conj;
  using size_type = typename extents<>::size_type;

  if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = j; i < A.extent(0); ++i) {
        A(i,j) += x(i) * conj(x(j));
      }
    }
  }
  else {
    for (size_type j = 0; j < A.extent(1); ++j) {
      for (size_type i = 0; i <= j; ++i) {
        A(i,j) += x(i) * conj(x(j));
      }
    }
  }
}

template<class ExecutionPolicy,
         class ElementType_x,
         extents<>::size_type ext_x,
         class Layout_x,
         class Accessor_x,
         class ElementType_A,
         extents<>::size_type numRows_A,
         extents<>::size_type numCols_A,
         class Layout_A,
         class Accessor_A,
         class Triangle>
void hermitian_matrix_rank_1_update(
  ExecutionPolicy&& /* exec */,
  std::experimental::basic_mdspan<ElementType_x, std::experimental::extents<ext_x>, Layout_x, Accessor_x> x,
  std::experimental::basic_mdspan<ElementType_A, std::experimental::extents<numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  Triangle t)
{
  hermitian_matrix_rank_1_update(x, A, t);
}

} // end namespace linalg
} // end inline namespace __p1673_version_0
} // end namespace experimental
} // end namespace std

#endif //LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_RANK_1_UPDATE_HPP_
