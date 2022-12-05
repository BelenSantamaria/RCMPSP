import numpy as np
import networkx as nx

from src.solution import Solution


def combined_heuristic(instance):

    max_time = np.sum(instance.durations)

    active_jobs = list()
    completed_jobs = list()
    scheduled_jobs = list()

    finishing_time = instance.n_jobs * [np.nan]
    available_resources = max_time * [instance.resource_availability]
    resource_transfers = np.zeros((instance.n_jobs, instance.n_jobs, instance.n_resources))
    resource_stocks = np.zeros((instance.n_jobs, instance.n_resources))

    # Represent instance as graph to compute critical paths
    G = nx.DiGraph()

    G.add_nodes_from(list(range(instance.n_jobs)))
    for i, suc in enumerate(instance.successors):
        for s in suc:
            G.add_edge(i, s)

    critical_paths = []
    for i in range(instance.n_jobs):
        if i == instance.n_jobs - 1:
            critical_paths.append(instance.durations[i])
        else:
            cp = max([sum(instance.durations[path]) for path in nx.all_simple_paths(G, i, instance.n_jobs - 1)])
            critical_paths.append(cp + instance.durations[i])
    critical_paths = np.array(critical_paths)

    t = 0

    completed_jobs.append(0)
    scheduled_jobs.append(0)
    finishing_time[0] = 0

    ## No está en el algoritmo original (pongo todos los recursos en el primer job aunque no los necesita porque tienen que estar en algún sitio)
    resource_stocks[0] = instance.resource_availability

    # Set of elegible jobs at t=0
    eligible_jobs = [
        i
        for i, pred in enumerate(instance.predecessors)
        if (set(pred).issubset(set(completed_jobs)) and i not in scheduled_jobs)
    ]

    # Iterate on increasing t until all jobs are assigned
    while len(scheduled_jobs) < instance.n_jobs:
        while len(eligible_jobs) > 0:
            # Apply job rule
            # Selección del job con menor duración del critical path
            selected_job = eligible_jobs[np.argmin(critical_paths[eligible_jobs])]

            supply_of_resources = list(np.zeros(instance.n_resources))
            transfer_feasible_jobs = [[], [], [], []]
            for r in range(instance.n_resources):
                # Compute transfer feasible set for each resource type
                transfer_feasible_jobs[r] = [
                    job
                    for job in scheduled_jobs
                    if ((resource_stocks[job][r] > 0) and (finishing_time[job] + instance.transfer_times[job, selected_job] < t))
                ]

                # Compute total supply of resources in t
                supply_of_resources[r] = np.sum([resource_stocks[job, r] for job in transfer_feasible_jobs[r]])
            # If supply is sufficient for resource
            if (supply_of_resources >= instance.required_resources[selected_job]).all():
                # find delivering jobs for all r
                for r in range(instance.n_resources):
                    # Repeat until resource demand is satisfied
                    while resource_stocks[selected_job, r] < instance.required_resources[selected_job, r]:
                        # Apply resource transfer rule
                        # minTT
                        ordered_transfer_jobs = instance.transfer_times[transfer_feasible_jobs[r], selected_job]
                        transfer_job = transfer_feasible_jobs[r][np.argmin(ordered_transfer_jobs)]
                        transfer_feasible_jobs[r].remove(transfer_job)

                        # Compute transferable amount of resource
                        resource_transfers[transfer_job, selected_job, r] = min(
                            resource_stocks[transfer_job, r],
                            instance.required_resources[selected_job][r] - resource_stocks[selected_job, r])

                        # Update resource stocks
                        resource_stocks[selected_job, r] += resource_transfers[transfer_job, selected_job, r]
                        resource_stocks[transfer_job, r] -= resource_transfers[transfer_job, selected_job, r]

            # Schedule selected_job
            finishing_time[selected_job] = t + instance.durations[selected_job]
            scheduled_jobs.append(selected_job)
            active_jobs.append(selected_job)

            # Reduce available capacity of resources
            if instance.durations[selected_job] > 0:
                available_resources[
                t: t + instance.durations[selected_job]
                ] -= instance.required_resources[selected_job]

            # Remove selected_job from elegible_jobs set
            eligible_jobs.remove(selected_job)

        # Increase current time
        t += 1

        if t >= max_time:
            return Solution(np.nan, np.nan, np.nan, np.nan)

        # Compute set os jobs just completed
        jobs_just_completed = [job for job in active_jobs if finishing_time[job] < t]

        # Update set of active and completed jobs
        for job in jobs_just_completed:
            completed_jobs.append(job)
            active_jobs.remove(job)

        # Compute set of elegible tasks
        eligible_jobs = [
            job
            for job in set(range(instance.n_jobs)).difference(set(scheduled_jobs))
            if set(instance.predecessors[job]).issubset(set(completed_jobs))
               and (instance.required_resources[job] <= available_resources[t]).all()
        ]

    return Solution(np.nan, max(finishing_time), finishing_time, [available_resources[:max(finishing_time)]])
