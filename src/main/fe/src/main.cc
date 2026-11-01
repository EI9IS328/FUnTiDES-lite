//************************************************************************
//  SEM proxy application v.0.0.1
//
//  main.cpp: this main file is simply a driver
//************************************************************************

#include "sem_proxy.h"
#include "sem_proxy_options.h"

#include <fstream>
#include <iomanip>

time_point<steady_clock> startInitTime;

static void saveMetrics(const SemProxyMetrics& metrics)
{
  std::ofstream output("metrics.json");
  if (!output) throw std::runtime_error("Cannot open metrics.json");
  // Un rapport par essai ; une nouvelle simulation réinitialise Python à null.
  output << std::setprecision(17)
         << "{\n  \"simulation\": {\n"
         << "    \"total_seconds\": " << metrics.total_seconds << ",\n"
         << "    \"compute_seconds\": " << metrics.compute_seconds << ",\n"
         << "    \"snapshot_write_seconds\": "
         << metrics.snapshot_write_seconds << ",\n"
         << "    \"sismogram_write_seconds\": "
         << metrics.sismogram_write_seconds << ",\n"
         << "    \"snapshot_mb\": " << metrics.snapshot_mb << ",\n"
         << "    \"sismogram_mb\": " << metrics.sismogram_mb << "\n"
         << "  },\n  \"postprocessing\": {\n"
         << "    \"snapshots\": {\"read_seconds\": null, "
         << "\"analysis_seconds\": null, \"visualization_seconds\": null},\n"
         << "    \"sismos\": {\"read_seconds\": null, "
         << "\"analysis_seconds\": null, \"visualization_seconds\": null}\n"
         << "  }\n}\n";
  output.close();
  if (!output) throw std::runtime_error("Cannot write metrics.json");
}

void compute(SEMproxy &semsim)
{
  cout << "\n+================================= " << endl;
  cout << "| Running SEM Application ...      " << endl;
  cout << "+================================= \n" << endl;

  // start timer
  time_point<steady_clock> startRunTime = steady_clock::now();
  semsim.run();

  cout << "\n+================================= " << endl;
  cout << "| SEM Application Finished.       " << endl;
  cout << "+================================= \n" << endl;

  // print timing information
  cout << "Elapsed Initial Time : "
       << duration<double>(startRunTime - startInitTime).count()
       << " seconds." << endl;
  cout << "Elapsed Run Time : "
       << duration<double>(steady_clock::now() - startRunTime).count()
       << " seconds."
       << endl;
};

void compute_loop(SEMproxy &semsim) { compute(semsim); }

int main(int argc, char *argv[])
{
  startInitTime = steady_clock::now();
  SemProxyMetrics metrics;

#ifdef USE_KOKKOS
  setenv("OMP_PROC_BIND", "spread", 1);
  setenv("OMP_PLACES", "threads", 1);
  Kokkos::initialize(argc, argv);
  {
#endif

    cxxopts::Options options("SEM Proxy", "Runs the SEM simulation.");
    options.allow_unrecognised_options();  // lets Kokkos flags pass

    options.add_options()("h,help", "Print help message");

    SemProxyOptions opt;
    SemProxyOptions::bind_cli(options, opt);

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    try
    {
      opt.validate();
    }
    catch (const std::exception &e)
    {
      // your error path (no help printing here)
      std::cerr << "Invalid options: " << e.what() << "\n";
      return 1;
    }

    cout << "+==================================+" << endl;
    cout << "| Initializing SEM Application ... |" << endl;
    cout << "+==================================+\n" << endl;

    SEMproxy semsim(opt);

    compute_loop(semsim);
    metrics = semsim.getMetrics();

#ifdef USE_KOKKOS
  }
  Kokkos::finalize();
#endif

  // Arrêter le temps total avant l'écriture du rapport de mesure lui-même.
  metrics.total_seconds =
      duration<double>(steady_clock::now() - startInitTime).count();
  saveMetrics(metrics);
  cout << "Elapsed TotalExe Time : "
       << metrics.total_seconds << " seconds.\n"
       << endl;
  return (0);
}
