import time
from collections import defaultdict

from paramiko import RSAKey
import os
from fabric import Connection
import re

from typing import List

IRIDIS_ADDRESS = "iridis5_a.soton.ac.uk"
USERID = "ii1g17"
IRIDIS_SSH_KEY = os.path.expanduser("/Users/ioan/.ssh/iridis5")

PARTION_TO_NODE = {"ecsstaff": ["alpha51", "alpha52", "alpha53"],
                   "ecsall": ["alpha54", "alpha55", "alpha56"],
                   "gpu": [f"indigo{i}" for i in range(51, 61)],
                   "gtx1080": [f"pink{i}" for i in range(51, 61)]}

PARTION_TO_NODE_NAME = {"ecsstaff": "alpha",
                        'gpu': 'indigo',
                        'gtx1080': 'pink'}

NODE_NAME_TO_PARTITION = {v: k for k, v in PARTION_TO_NODE_NAME.items()}

NODE_MAX_RES = {"alpha": {"CPU": 64, "RAM": 371, "GPU": 4},
                "indigo": {"CPU": 40, "RAM": 187.5, "GPU": 2},
                "pink": {"CPU": 56, "RAM": 125, "GPU": 4}}

PARTITION_MAX_RES = {"ecsstaff": {"CPU": 64 * 3, "RAM": 371 * 3, "GPU": 4 * 3},
                     "ecsall": {"CPU": 64 * 3, "RAM": 371 * 3, "GPU": 4 * 3},
                     "gpu": {"CPU": 40 * 10, "RAM": 187.5 * 10, "GPU": 2 * 10},
                     "gtx1080": {"CPU": 56 * 10, "RAM": 125 * 10, "GPU": 4 * 10}}


class RemoteConnectionManager:
    def __init__(self):
        self.private_key = RSAKey(filename=IRIDIS_SSH_KEY)
        self.connection = self._establish_connection()

    def _establish_connection(self):
        return Connection(host=IRIDIS_ADDRESS, user=USERID, connect_kwargs={'pkey': self.private_key})

    def run_command(self, command):
        try:
            if not self.connection.is_connected:
                self.connection = self._establish_connection()
            return self.connection.run(command, hide=True).stdout
        except Exception as e:
            print(f"Error: {e}")
            return ""


def parse_scontrol_output(output: str) -> dict:
    return dict(pair.split("=", maxsplit=1) for line in output.split('\n')
                for pair in re.split(' ', line) if "=" in pair)


def memory_string_to_GB(memory_string: str) -> float:
    """Converts the memory string to GB"""
    memory_string = memory_string.strip().upper()
    multipliers = {'G': 1, 'M': 1e-3, 'K': 1e-6, 'T': 1e3}

    for unit, multiplier in multipliers.items():
        if unit in memory_string:
            return round(float(memory_string.replace(unit, '')) * multiplier, 1)

    # Default to MB if no unit is specified
    return round(float(memory_string) * 1e-3, 1)


# -------------------------------------------Node Info-------------------------------------------
def get_node_info(node_name: str, conn_manager: RemoteConnectionManager) -> dict:
    """
     Gets the GPU, RAM, CPU etc. information for a node.
     Args:
         node_name: The name of the node to check.
     Returns:
         dict: A dictionary containing the node's information, property->value.
     """
    output = conn_manager.run_command(f"scontrol show node {node_name}")
    return parse_scontrol_output(output)


def get_node_GPU_name(node_info: dict) -> str:
    return node_info["Gres"].split(":")[1]


def get_node_GPU_count(node_info: dict) -> int:
    return int(node_info["Gres"].split(":")[2])


def get_node_CPU_cores_count(node_info: dict) -> int:
    return int(node_info["CPUTot"])


def get_node_RAM_count(node_info: dict) -> float:
    """Returns the total RAM available in GB"""
    return memory_string_to_GB(node_info["RealMemory"])


def get_node_allocated_GPU(node_info: dict, locked_usage: bool = True) -> int:
    """For gtx1080 and gpu partitions, the GPU resources are locked to the user,
    so we can return the number of GPUs locked or used"""

    node_name = re.sub(r'\d', '', node_info["NodeName"])
    if not locked_usage or node_name == "alpha":
        if "gres/gpu=" in node_info["AllocTRES"]:
            return int(node_info["AllocTRES"].split("gres/gpu=")[1])
    else:
        # if there is no owner (user locked to resources) that means the node is free
        if NODE_NAME_TO_PARTITION[node_name] in ["gtx1080", "gpu"] and node_info["Owner"].strip() != 'N/A':
            return NODE_MAX_RES[node_name]["GPU"]
    return 0


def get_node_allocated_CPU_cores(node_info: dict) -> int:
    return int(node_info["CPUAlloc"])


def get_node_allocated_RAM(node_info: dict) -> float:
    """Returns the allocated RAM in GB"""
    return memory_string_to_GB(node_info["AllocMem"])


def get_node_free_RAM(node_info: dict) -> float:
    """Returns the free RAM in GB"""
    if "N/A" in node_info["FreeMem"]:
        return 0

    return memory_string_to_GB(node_info["FreeMem"])


def get_node_current_power_usage(node_info: dict) -> float:
    return float(node_info["CurrentWatts"])


def get_node_avg_power_usage(node_info: dict) -> float:
    return float(node_info["AverageWatts"])


def get_node_state(node_name: str, conn_manager: RemoteConnectionManager) -> str:
    return conn_manager.run_command(f'sinfo --noheader -n {node_name} --Format StateLong').strip()


# -------------------------------------------Job Info-----------------------------------------------
def get_job_info(job_info: str, conn_manager: RemoteConnectionManager) -> dict:
    """
     Gets the all the information about a job and returns it as a dictionary.
     Args:
         node_name: The name of the job to check.
     Returns:
         dict: A dictionary containing the job's information, property->value.
     """
    output = conn_manager.run_command(f"scontrol show job {job_info}")
    return parse_scontrol_output(output)


def get_job_allocated_GPU(job_info: dict) -> int:
    if "gres/gpu=" not in job_info["TRES"]:
        return 0
    return int(job_info["TRES"].split("gres/gpu=")[1])


# -------------------------------------------User Info----------------------------------------------

def get_user_allocated_nodes_on_partition(partition_name: str,
                                          conn_manager: RemoteConnectionManager,
                                          username: str) -> List[str]:
    nodes = conn_manager.run_command(f"sacct --noheader --partition={partition_name} -u {username} "
                                     f"--state=RUNNING -X --format NodeList")
    return [el for el in set([n.strip() for n in nodes.strip().split("\n")]) if el != ""]


def get_user_job_ids_on_partition(partition_name: str,
                                  conn_manager: RemoteConnectionManager,
                                  username: str,
                                  job_state: str = "RUNNING") -> List[str]:
    output = conn_manager.run_command(f"squeue --noheader -p {partition_name} -u {username} -t {job_state} -o %A")
    return output.strip().split("\n")


def get_user_allocated_CPU_cores_on_partition(partition_name: str,
                                              conn_manager: RemoteConnectionManager,
                                              username: str) -> int:
    cpus = conn_manager.run_command(f"sacct --noheader --partition={partition_name} "
                                    f"-u {username} --state=RUNNING  -X --format AllocCPUs")
    if cpus.strip() == "":
        return 0
    return sum([int(cpu.strip()) for cpu in cpus.strip().split("\n")])


def get_user_allocated_GPU_on_partition(partition_name: str,
                                        conn_manager: RemoteConnectionManager,
                                        username: str,
                                        locked_usage: bool) -> int:
    """I can't find a nice way of getting th GPU count. AllocTRES doesn't really return what we want.
    However, the gpu and gtx1080 partitions lock the GPU resources on the node to the user,
    regardless of the number of GPUs requested.
    So, we can just count the number of nodes the user has allocated and multiply by the number of GPUs per node.

    For ecsstaff and ecsall, we can't do this, so we just have to go through all the running jobs.
    """
    if partition_name in ["gpu", "gtx1080"] and locked_usage:
        return len(get_user_allocated_nodes_on_partition(partition_name, conn_manager, username)) * \
            NODE_MAX_RES[PARTION_TO_NODE_NAME[partition_name]]["GPU"]
    else:
        running_jobs = get_user_job_ids_on_partition(partition_name, conn_manager, username)
        return sum([get_job_allocated_GPU(get_job_info(job_id, conn_manager)) for job_id in running_jobs])


def get_user_allocated_RAM_on_partition(partition_name: str,
                                        conn_manager: RemoteConnectionManager,
                                        username: str) -> int:
    """Returns the RAM usage in GB"""
    rams = conn_manager.run_command(f"sacct --noheader --partition={partition_name} "
                                    f"-u {username} --state=RUNNING  -X --format ReqMem")
    if rams.strip() == "":
        return 0
    return round(sum([memory_string_to_GB(r.strip()) for r in rams.strip().split("\n")]), 1)


# -------------------------------------------Partition Info-------------------------------------------

def get_partition_queue_job_info(partition_name: str,
                                 conn_manager: RemoteConnectionManager,
                                 attribute: str,
                                 job_state: str = "RUNNING"):
    output = conn_manager.run_command(f'squeue --noheader -p {partition_name} -t {job_state} --Format {attribute}')
    return [o.strip() for o in output.strip().split("\n")]


def get_partition_CPU_cores_info(partition_name: str,
                                 conn_manager: RemoteConnectionManager,
                                 resource_state: str) -> int:
    cpu_info = conn_manager.run_command(f'sinfo --noheader -p {partition_name} --Format CPUsState')

    allocated, idle, other, total = map(int, [c for c in cpu_info.strip().split("/") if c != ""])

    resource_state_map = {
        "ALLOCATED": allocated,
        "IDLE": idle,
        "OTHER": other,
        "TOTAL": total
    }
    return resource_state_map.get(resource_state.upper(), ValueError(f"Invalid resource state: {resource_state}"))


def get_partition_allocated_GPU(partition_name: str,
                                conn_manager: RemoteConnectionManager,
                                locked_usage: bool = False) -> int:
    """Can't find a way to get this number directly, so we are iterating through all the nodes"""
    return sum([get_node_allocated_GPU(get_node_info(node, conn_manager), locked_usage=locked_usage)
                for node in PARTION_TO_NODE[partition_name]])


def get_partition_allocated_RAM(partition_name: str,
                                conn_manager: RemoteConnectionManager) -> float:
    """Returns the RAM usage in GB"""
    rams = conn_manager.run_command(f"sacct --noheader --partition={partition_name} "
                                    f"--allusers --state=RUNNING  -X --format ReqMem")
    return sum([memory_string_to_GB(r) for r in rams.strip().split("\n")])


# --------------------------------------------info gathering-----------------------------------------

def calculate_percentage(numerator, denominator):
    return round((numerator / denominator) * 100, 1)


def aggregate_partition_info(conn_manager: RemoteConnectionManager, locked_usage: bool = True) -> dict:
    """Gets all the information about all GPU partitions and returns it as a dictionary.
    This is needed to populate the front end"""

    info = defaultdict(dict)
    for p in PARTION_TO_NODE.keys():

        info[p] = {'cpu_free': get_partition_CPU_cores_info(p, conn_manager, "idle"),
                   'cpu_total': get_partition_CPU_cores_info(p, conn_manager, "total"),
                   'ram_free': max(round((PARTITION_MAX_RES[p]['RAM'] - get_partition_allocated_RAM(p, conn_manager)), 1), 0),
                   'ram_total': PARTITION_MAX_RES[p]['RAM'],

                   'gpu_name': get_node_GPU_name(get_node_info(PARTION_TO_NODE[p][0], conn_manager)),
                   'gpu_free': PARTITION_MAX_RES[p]['GPU'] - get_partition_allocated_GPU(p, conn_manager, locked_usage),
                   'gpu_total': PARTITION_MAX_RES[p]['GPU'],
                   'nodes': {}
                   }

        info[p]['cpu_free_percentage'] = calculate_percentage(info[p]['cpu_free'], info[p]['cpu_total'])
        info[p]['ram_free_percentage'] = calculate_percentage(info[p]['ram_free'], info[p]['ram_total'])
        info[p]['gpu_free_percentage'] = calculate_percentage(info[p]['gpu_free'], info[p]['gpu_total'])

        for n in PARTION_TO_NODE[p]:
            node_info = get_node_info(n, conn_manager)
            node_name = "".join(x for x in n if x.isalpha())
            info[p]['nodes'][n] = {'state': get_node_state(n, conn_manager),
                                   'cpu_allocated': get_node_allocated_CPU_cores(node_info),
                                   'cpu_total': NODE_MAX_RES[node_name]['CPU'],
                                   # for gtx1080ti and gpu partitions, Alloc ram always returns 0??
                                   'ram_allocated': get_node_allocated_RAM(node_info) if p in ['ecsstaff',
                                                                                               'ecsall'] else
                                   round(NODE_MAX_RES[node_name]['RAM'] - get_node_free_RAM(node_info), 1),
                                   'ram_total': NODE_MAX_RES[node_name]['RAM'],
                                   'gpu_allocated': get_node_allocated_GPU(node_info, locked_usage=locked_usage),
                                   'gpu_total': NODE_MAX_RES[node_name]['GPU']
                                   }
            info[p]['nodes'][n]['cpu_allocated_percentage'] = calculate_percentage(info[p]['nodes'][n]['cpu_allocated'],
                                                                                   info[p]['nodes'][n]['cpu_total'])
            info[p]['nodes'][n]['ram_allocated_percentage'] = calculate_percentage(info[p]['nodes'][n]['ram_allocated'],
                                                                                   info[p]['nodes'][n]['ram_total'])
            info[p]['nodes'][n]['gpu_allocated_percentage'] = calculate_percentage(info[p]['nodes'][n]['gpu_allocated'],
                                                                                   info[p]['nodes'][n]['gpu_total'])

    return info


def aggregate_user_info(conn_manager: RemoteConnectionManager) -> dict:
    """gets the ram, cpu and gpu usage for each user that has a running job on the gpu partitions
    """

    info = defaultdict(lambda: {'cpu_allocated': 0, 'ram_allocated': 0, 'gpu_allocated': 0, 'gpu_locked': 0})
    user_running_set = set()

    # get all the users that have running jobs on the cluster
    for partition in PARTION_TO_NODE.keys():
        user_queue_job_info = get_partition_queue_job_info(partition, conn_manager, "UserName")
        user_running_set.update(user_queue_job_info)

    for user in user_running_set:
        for partition in PARTION_TO_NODE.keys():
            info[user]['cpu_allocated'] += get_user_allocated_CPU_cores_on_partition(partition, conn_manager, user)
            info[user]['ram_allocated'] += get_user_allocated_RAM_on_partition(partition, conn_manager, user)
            info[user]['gpu_allocated'] += get_user_allocated_GPU_on_partition(partition, conn_manager, user, False)
            info[user]['gpu_locked'] += get_user_allocated_GPU_on_partition(partition, conn_manager, user, True)

    # apply round to RAM values with 1 decimal place
    for user in info.keys():
        info[user]['ram_allocated'] = round(info[user]['ram_allocated'], 1)

    return info


def filter_users_with_no_GPU_usage(info: dict) -> dict:
    """Keeps only the bad users"""
    return {user: user_info for user, user_info in info.items() if user_info['gpu_allocated'] == 0}


def filter_users_with_partial_GPU_usage(info: dict) -> dict:
    """Keeps only the bad users"""
    return {user: user_info for user, user_info in info.items() if user_info['gpu_allocated'] < user_info['gpu_locked']
            and user_info['gpu_allocated'] != 0}


if __name__ == "__main__":
    conn_manager = RemoteConnectionManager()
    import pprint

    while True:
        node_info = get_node_info("pink51", conn_manager)
        a = aggregate_user_info(conn_manager)
        pprint.pprint(a)
        time.sleep(1)  # Wait for a minute before checking again
