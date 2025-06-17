import os
import shutil
import pickle
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import scripts.SumoScripts as SS

SIM_NAME = "sumomaps"
sumo_tools_path = os.environ.get("SUMOTOOLSPATH", "")

if not sumo_tools_path:
    raise EnvironmentError("Environment variable 'SUMOTOOLSPATH' is not set or not accessible.")

command = (
    f'python "{os.path.join(sumo_tools_path, "randomTrips.py")}" '
    '--random -n map.net.xml -b 0 -e 1 -p 0.002 --intermediate 250 '
    '--validate'
)

CWD_PATH = os.getcwd()
END_TIME = 3000
T0 = 300  # time to start collecting edge load data

def run_simulation(network):
    try:
        print(f"[{network}] Starting...")
        simName = network.split('.')[0]
        targetName = os.path.join(CWD_PATH, SIM_NAME, network)

        os.makedirs(targetName, exist_ok=True)
        os.chdir(targetName)

        # Generate trips
        os.system(command)

        # Generate routes
        os.system('duarouter -n map.net.xml --route-files trips.trips.xml -o map.rou.xml --ignore-errors')

        # Generate config and additional file
        SS.generateCFG(END_TIME)
        SS.generateAdditionalFile()

        # Run SUMO simulation
        os.system('sumo -c map.sumo.cfg -v --additional-files additional.xml --fcd-output sumoTace.xml')

        # Extract simulation data
        G, idToEdgeMap = SS.sumoNetToNetworkx("map.net.xml")
        simulations = SS.edgeDataForNetworkx("edgeaddout.xml", T0, idToEdgeMap, G)

        G = simulations[0]
        C = [G.edges[edge]['capacity'] for edge in G.edges]

        with open('G.pkl', 'wb') as f:
            pickle.dump(G, f)
        with open('C.pkl', 'wb') as f:
            pickle.dump(C, f)

        simLoads = [[sim.edges[edge]['load'] for edge in sim.edges] for sim in simulations]
        with open('W.pkl', 'wb') as f:
            pickle.dump(simLoads, f)

        # Clean up
        keep_files = {'C.pkl', 'G.pkl', 'W.pkl'}
        for fname in os.listdir():
            if fname not in keep_files:
                try:
                    if os.path.isfile(fname):
                        os.remove(fname)
                    elif os.path.isdir(fname):
                        shutil.rmtree(fname)
                except Exception as e:
                    print(f"Failed to delete {fname}: {e}")

    except Exception as e:
        print(f"Error while processing {network}: {e}")
    finally:
        os.chdir(CWD_PATH)

def parallel_run(networks):
    with Pool(processes=4) as pool:
        list(tqdm(pool.imap_unordered(run_simulation, networks), total=len(networks)))

if __name__ == '__main__':
    NETWORKS = os.listdir(SIM_NAME)
    parallel_run(NETWORKS)
