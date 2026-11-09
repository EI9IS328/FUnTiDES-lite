#pragma once

#include <data_type.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <stdexcept>

// Extrema over all nodes of one pressure-field time buffer.
struct PressureExtrema
{
  double minimum_pressure;
  double maximum_pressure;
  double maximum_absolute_pressure;
};

template <class PressureArray>
inline void validatePressureField(const PressureArray& pressure, int timeIndex)
{
  // timeIndex selects a stored time buffer, not the simulation step number.
  if (pressure.extent(0) == 0)
    throw std::invalid_argument("Pressure field must contain at least one node");
  if (timeIndex < 0 ||
      static_cast<std::size_t>(timeIndex) >= pressure.extent(1))
    throw std::invalid_argument("Pressure time index is out of range");
}

// Host traversal for Vector storage or host-accessible Kokkos views.
template <class PressureArray>
inline PressureExtrema computePressureExtremaCPU(const PressureArray& pressure,
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

#ifdef USE_KOKKOS
// Experimental timings are independent of the full snapshot-analysis timer.
struct MeasuredExtrema
{
  PressureExtrema values{};
  double prepare_seconds = 0, compute_seconds = 0, result_seconds = 0, total_seconds = 0;
  std::size_t input_bytes = 0, result_bytes = 0;
};

class PressureExtremaWorkspace
{
  using Space = Kokkos::DefaultExecutionSpace::memory_space;
  using MinMax = Kokkos::MinMax<double, Space>;
  using Finite = Kokkos::LAnd<int, Space>;
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace> host_field_;
  typename MinMax::result_view_type device_extrema_;
  typename Finite::result_view_type device_finite_;
  Kokkos::View<typename MinMax::value_type, Kokkos::HostSpace> host_extrema_;
  Kokkos::View<int, Kokkos::HostSpace> host_finite_;
  bool on_cpu_;

 public:
  PressureExtremaWorkspace(std::size_t nodes, bool on_cpu) : on_cpu_(on_cpu)
  {
    // Allocate once before the time loop, never inside the measured block.
    if (on_cpu_)
      host_field_ = decltype(host_field_)("extrema_host_field", nodes, 1);
    else
    {
      device_extrema_ = typename MinMax::result_view_type("extrema_device_result");
      device_finite_ = typename Finite::result_view_type("extrema_device_finite");
      host_extrema_ = decltype(host_extrema_)("extrema_host_result");
      host_finite_ = decltype(host_finite_)("extrema_host_finite");
    }
  }

  MeasuredExtrema measure(const arrayReal& pressure, int time_index)
  {
    validatePressureField(pressure, time_index);
    using Clock = std::chrono::steady_clock;
    const auto seconds = [](auto start) {
      return std::chrono::duration<double>(Clock::now() - start).count();
    };
    // Exclude outstanding physics kernels from this statistical measurement.
    Kokkos::fence();
    const auto total_start = Clock::now();
    MeasuredExtrema result;
    if (on_cpu_)
    {
      const auto prepare_start = Clock::now();
      Kokkos::deep_copy(Kokkos::subview(host_field_, Kokkos::ALL(), 0),
                        Kokkos::subview(pressure, Kokkos::ALL(), time_index));
      Kokkos::fence();
      result.prepare_seconds = seconds(prepare_start);
      result.input_bytes = pressure.extent(0) * sizeof(float);
      const auto compute_start = Clock::now();
      result.values = computePressureExtremaCPU(host_field_, 0);
      result.compute_seconds = seconds(compute_start);
    }
    else
    {
      const auto field = pressure;
      const auto compute_start = Clock::now();
      // Device result views prevent an implicit host-result copy inside this timer.
      Kokkos::parallel_reduce(
          "MeasuredPressureExtrema",
          Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, pressure.extent(0)),
          KOKKOS_LAMBDA(const std::size_t node, typename MinMax::value_type& extrema,
                        int& finite) {
            const double value = field(node, time_index);
            if (!Kokkos::isfinite(value)) finite = 0;
            else
            {
              if (value < extrema.min_val) extrema.min_val = value;
              if (value > extrema.max_val) extrema.max_val = value;
            }
          }, MinMax(device_extrema_), Finite(device_finite_));
      Kokkos::fence();
      result.compute_seconds = seconds(compute_start);
      const auto result_start = Clock::now();
      Kokkos::deep_copy(host_extrema_, device_extrema_);
      Kokkos::deep_copy(host_finite_, device_finite_);
      Kokkos::fence();
      if (!host_finite_()) throw std::invalid_argument("Pressure must be finite");
      const auto extrema = host_extrema_();
      result.values = {extrema.min_val, extrema.max_val,
                      std::max(std::abs(extrema.min_val), std::abs(extrema.max_val))};
      result.result_seconds = seconds(result_start);
      result.result_bytes = sizeof(typename MinMax::value_type) + sizeof(int);
    }
    result.total_seconds = seconds(total_start);
    return result;
  }
};
#endif

struct PressureMeanVariance
{
  double mean_pressure;
  double pressure_variance;
};

// Constant-size state, reusable for field nodes or successive receiver samples.
struct WelfordState
{
  std::size_t count;
  double mean;
  double m2;  // Corrected sum of squared deviations; population variance is m2/count.
  bool all_finite;

  PROXY_HOST_DEVICE
  WelfordState() : count(0), mean(0), m2(0), all_finite(true) {}

  PROXY_HOST_DEVICE
  void update(double value)
  {
#ifdef USE_KOKKOS
    const bool finite = Kokkos::isfinite(value);
#else
    const bool finite = std::isfinite(value);
#endif
    // Record invalid input instead of throwing from code that can run on a GPU.
    if (!finite)
    {
      all_finite = false;
      return;
    }

    ++count;
    const double delta = value - mean;
    mean += delta / static_cast<double>(count);
    m2 += delta * (value - mean);
  }

  PROXY_HOST_DEVICE
  void merge(const WelfordState& other)
  {
    const bool finite = all_finite && other.all_finite;
    // An empty state is the reduction identity; retain any invalid-input flag.
    if (other.count == 0)
    {
      all_finite = finite;
      return;
    }
    if (count == 0)
    {
      *this = other;
      all_finite = finite;
      return;
    }

    const std::size_t combined_count = count + other.count;
    const double delta = other.mean - mean;
    const double other_weight =
        static_cast<double>(other.count) / static_cast<double>(combined_count);
    // Account for the different group means when combining partial variances.
    m2 += other.m2 + delta * delta * static_cast<double>(count) * other_weight;
    mean += delta * other_weight;
    count = combined_count;
    all_finite = finite;
  }
};

// Host-side validation and final division, matching NumPy's ddof=0 convention.
inline PressureMeanVariance finalizePressureMeanVariance(const WelfordState& state)
{
  if (!state.all_finite)
    throw std::invalid_argument("Pressure must be finite");
  if (state.count == 0)
    throw std::invalid_argument("No pressure samples have been received");
  return {state.mean, state.m2 / static_cast<double>(state.count)};
}

inline PressureMeanVariance computePressureMeanVarianceCPU(const arrayReal& pressure,
                                                         int timeIndex)
{
  validatePressureField(pressure, timeIndex);
#ifdef USE_KOKKOS
  Kokkos::fence();  // Complete pending kernels before traversing the field on the host.
#endif
  WelfordState state;
  for (std::size_t node = 0; node < pressure.extent(0); ++node)
    state.update(pressure(node, timeIndex));
  return finalizePressureMeanVariance(state);
}

#ifdef USE_KOKKOS
// Custom reduction: each worker updates a local state, then Kokkos joins the states.
struct PressureWelfordReduction
{
  using value_type = WelfordState;

  arrayReal field;
  int time_index;

  PROXY_HOST_DEVICE
  void init(value_type& state) const { state = value_type(); }

  PROXY_HOST_DEVICE
  void join(value_type& destination, const value_type& source) const
  {
    destination.merge(source);
  }

  PROXY_HOST_DEVICE
  void operator()(const std::size_t node, value_type& state) const
  {
    state.update(field(node, time_index));
  }
};

inline PressureMeanVariance computePressureMeanVarianceKokkos(const arrayReal& pressure,
                                                            int timeIndex)
{
  validatePressureField(pressure, timeIndex);
  WelfordState state;
  Kokkos::parallel_reduce(
      "PressureMeanVariance",
      Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, pressure.extent(0)),
      PressureWelfordReduction{pressure, timeIndex}, state);
  Kokkos::fence();  // Validate and finalize the reduced state on the host.
  return finalizePressureMeanVariance(state);
}
#endif  // USE_KOKKOS
