
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_IDX_ABS_MAX_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL_P1673_BITS_KOKKOSKERNELS_IDX_ABS_MAX_HPP_

#include <KokkosBlas1_iamax.hpp>

namespace KokkosKernelsSTD {

// keeping this in mind: https://github.com/kokkos/stdBLAS/issues/122

template<class ExeSpace,
         class ElementType,
         std::experimental::extents<>::size_type ext0,
         class Layout>
std::experimental::extents<>::size_type
idx_abs_max(kokkos_exec<ExeSpace>,
	    std::experimental::mdspan<
	    ElementType,
	    std::experimental::extents<ext0>,
	    Layout,
	    std::experimental::default_accessor<ElementType>> v)
{
  // note that -1 here, this is related to:
  // https://github.com/kokkos/stdBLAS/issues/114

#if defined LINALG_ENABLE_TESTS
  std::cout << "idx_abs_max: kokkos impl\n";
#endif

  return KokkosBlas::iamax(Impl::mdspan_to_view(v)) - 1;
}

}
#endif
