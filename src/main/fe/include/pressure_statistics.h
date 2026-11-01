#pragma once

#include <data_type.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Extrema over all nodes of one pressure-field time buffer.
struct PressureExtrema
{
  double minimum_pressure;
  double maximum_pressure;
  double maximum_absolute_pressure;
};

inline void validatePressureField(const arrayReal& pressure, int timeIndex)
{
  // timeIndex selects a stored time buffer, not the simulation step number.
  if (pressure.extent(0) == 0)
    throw std::invalid_argument("Pressure field must contain at least one node");
  if (timeIndex < 0 ||
      static_cast<std::size_t>(timeIndex) >= pressure.extent(1))
    throw std::invalid_argument("Pressure time index is out of range");
}

// Host traversal for Vector storage or host-accessible Kokkos views.
inline PressureExtrema computePressureExtremaCPU(const arrayReal& pressure,
                                                 int timeIndex)
{
  validatePressureField(pressure, timeIndex);
#ifdef USE_KOKKOS
  Kokkos::fence();  // Wait for pending kernels before accessing pressure on the host.
#endif

  // Start from an actual value: the field may be entirely positive or negative.
  double minimum = pressure(0, timeIndex);
  double maximum = minimum;
  for (std::size_t node = 0; node < pressure.extent(0); ++node)
    {
      const double value = pressure(node, timeIndex);
      if (!std::isfinite(value))
        throw std::invalid_argument("Pressure must be finite");
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
  // The largest absolute value is one of the two extrema; no extra scan is needed.
  return {minimum, maximum, std::max(std::abs(minimum), std::abs(maximum))};
}

// This implementation is compiled only when the project enables Kokkos.
#ifdef USE_KOKKOS
// Parallel traversal on the configured Kokkos backend (OpenMP or CUDA).
inline PressureExtrema computePressureExtremaKokkos(const arrayReal& pressure,
                                                    int timeIndex)
{
  validatePressureField(pressure, timeIndex);
  using MinMax = Kokkos::MinMax<double>;
  // The reducers initialize these outputs and the corresponding local states.
  MinMax::value_type extrema;
  int all_finite;
  const auto field = pressure;  // Shallow View copy for capture; no field data is copied.

  // Reduce min/max and a finite-value flag together in a single field traversal.
  Kokkos::parallel_reduce(
                          "PressureExtrema",
                          Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, pressure.extent(0)),
                          KOKKOS_LAMBDA(const std::size_t node, MinMax::value_type& local_extrema,
                                        int& local_all_finite) {
                            const double value = field(node, timeIndex);
                            if (!Kokkos::isfinite(value))
                              {
                                local_all_finite = 0;
                              }
                            else
                              {
                                if (value < local_extrema.min_val) local_extrema.min_val = value;
                                if (value > local_extrema.max_val) local_extrema.max_val = value;
                              }
                          },
                          MinMax(extrema), Kokkos::LAnd<int>(all_finite));
  Kokkos::fence();  // Explicit synchronization before reading the results on the host.

  // Reject invalid values on the host: exceptions must not be thrown in the kernel.
  if (!all_finite)
    throw std::invalid_argument("Pressure must be finite");

  return {extrema.min_val, extrema.max_val,
          std::max(std::abs(extrema.min_val), std::abs(extrema.max_val))};
}
#endif  // USE_KOKKOS
