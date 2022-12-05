import numpy as np

from src.solution import Solution


def heuristic(instance):

    max_time = np.sum(instance.durations)

    active_jobs = set()
    completed_jobs = set()
    scheduled_jobs = set()

    finishing_time = instance.n_jobs * [np.nan]
    available_resources = max_time * [instance.resource_availability]
    resource_transfers = np.zeros((instance.n_jobs, instance.n_jobs, instance.n_resources))
    resource_stocks = np.zeros((instance.n_jobs, instance.n_resources))

    t = 0

    completed_jobs.add(0)
    scheduled_jobs.add(0)
    finishing_time[0] = 0

    ## No está en el algoritmo original (pongo todos los recursos en el primer job aunque no los necesita porque tienen que estar en algún sitio)
    resource_stocks[0] = instance.resource_availability

    # Set of elegible jobs at t=0
    eligible_jobs = [
        i
        for i, pred in enumerate(instance.predecessors)
        if set(pred).issubset(completed_jobs)
    ]

    # Iterate on increasing t until all jobs are assigned
    while len(scheduled_jobs) < instance.n_jobs:
        while len(eligible_jobs) > 0:
            # Apply job rule

            # Selección del job con mayor duración de suma de sucesores
            succesors_durations = [instance.durations[instance.successors[i]].sum() + instance.durations[i] for i in eligible_jobs]
            selected_job = eligible_jobs[np.argmax(succesors_durations)]

            # Selección del job con menor duración
            # selected_job = eligible_jobs[np.argmin(instance.durations[eligible_jobs])]

            # Compute transfer feasible set for each resource type
            transfer_feasible_jobs = [
                (resource_stocks[job] > 0)
                * (finishing_time[job] + instance.transfer_times[job, selected_job] < t)
                for job in scheduled_jobs
            ]
            # Compute total supply os resources in t
            supply_of_resources = np.sum(
                transfer_feasible_jobs * resource_stocks[list(scheduled_jobs)], axis=0
            )

            # If supply is sufficient for all resources
            if (supply_of_resources >= instance.required_resources[selected_job]).all():
                # Repeat until resource demand is satisfied
                while (
                        resource_stocks[selected_job]
                        < instance.required_resources[selected_job]
                ).any():
                    # Apply resource transfer rule (min GAP)

                    missing_resources = instance.required_resources[selected_job] - resource_stocks[selected_job]

                    resource_needed = np.argmax(missing_resources)

                    feasible_jobs = [
                        s
                        for i, s in enumerate(scheduled_jobs)
                        if (transfer_feasible_jobs[i] * resource_stocks[s])[resource_needed]
                    ]
                    ordered_transfer_jobs = t - (
                            np.array(finishing_time)[feasible_jobs]
                            + instance.transfer_times[feasible_jobs, selected_job]
                    )

                    # Backward selection of transfer_job

                    transfer_job = None

                    for j in np.argsort(ordered_transfer_jobs):
                        if missing_resources[resource_needed] >= resource_stocks[feasible_jobs[j]][resource_needed]:
                            transfer_job = feasible_jobs[j]
                            break

                    if not transfer_job:
                        transfer_job = feasible_jobs[np.argmax([resource_stocks[j][resource_needed] for j in feasible_jobs])]

                    # Forward selection of transer_job

                    # transfer_job = feasible_jobs[np.argmin(ordered_transfer_jobs)]

                    # Compute transferable amount of resource
                    resource_transfers[transfer_job, selected_job, :] = np.min(
                        [
                            resource_stocks[transfer_job],
                            instance.required_resources[selected_job]
                            - resource_stocks[selected_job],
                        ],
                        axis=0,
                    )
                    # Update resource stocks
                    resource_stocks[selected_job] += resource_transfers[
                                                     transfer_job, selected_job, :
                                                     ]
                    resource_stocks[transfer_job] -= resource_transfers[
                                                     transfer_job, selected_job, :
                                                     ]

                # Schedule selected_job
                finishing_time[selected_job] = t + instance.durations[selected_job]
                scheduled_jobs.add(selected_job)
                active_jobs.add(selected_job)

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
            completed_jobs.add(job)
            active_jobs.remove(job)

        # Compute set of elegible tasks
        eligible_jobs = [
            job
            for job in set(range(instance.n_jobs)).difference(scheduled_jobs)
            if set(instance.predecessors[job]).issubset(completed_jobs)
               and (instance.required_resources[job] <= available_resources[t]).all()
        ]

    critical_paths_durations = []

    for job in range(1, instance.n_jobs):
        path_duration = instance.durations[instance.successors[job]].sum() + instance.durations[
            instance.predecessors[job]].sum() + instance.durations[job]
        critical_paths_durations.append(path_duration)

    mpdi = max(critical_paths_durations) - max(finishing_time)

    return Solution(mpdi, max(finishing_time), finishing_time, [available_resources[:max(finishing_time)]])
