import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass(frozen=False)
class SingleProject:
    n_jobs: int
    n_resources: int

    durations: np.ndarray  # job durations
    successors: List[List[int]]  # job successors
    predecessors: List[List[int]]  # job predecessors
    required_resources: np.ndarray  # job required resources
    resource_availability: np.ndarray  # resource capacities

    @classmethod
    def read_instance(cls, path: str):
        """
        Reads an instance of the RCSP from a file.
        Assumes the data is in the Patterson format.
        """
        with open(path) as f:
            lines = f.readlines()

        n_jobs, n_resources = list(map(int, lines[0].split()))
        resource_availability = list(map(int, lines[2].split()))

        durations = list()
        required_resources = list()
        n_successors = list()
        successors = list()

        for line in lines[4:]:
            d, *r = list(map(int, line.split()[: n_resources + 1]))
            n, *s = list(map(int, line.split()[n_resources + 1 :]))

            durations.append(d)
            required_resources.append(r)
            n_successors.append(n)
            successors.append(s)

        predecessors = [[] for _ in successors]
        for i, succ in enumerate(successors):
            for s in succ:
                predecessors[s - 1].append(i + 1)

        return SingleProject(
            n_jobs,
            n_resources,
            np.array(durations),
            successors,
            predecessors,
            np.array(required_resources),
            np.array(resource_availability),
        )


@dataclass(frozen=True)
class MultiProject:
    n_jobs: int
    n_resources: int

    durations: np.ndarray  # job durations
    successors: List[np.ndarray]  # job successors
    predecessors: List[np.ndarray]  # job predecessors
    required_resources: np.ndarray  # job required resources
    resource_availability: np.ndarray  # resource capacities
    transfer_times: np.ndarray  # transfer times

    @classmethod
    def unify_projects(cls, single_projects: List[SingleProject]):
        """
        Combines several single projects into one multi-project
        """

        n_jobs = sum([project.n_jobs for project in single_projects]) + 2

        # Create dummy resources to make sure all projects have the same number
        n_resources = max([project.n_resources for project in single_projects])

        durations = np.concatenate([project.durations for project in single_projects])
        durations = list([0]) + list(durations) + list([0])

        previous_jobs = 0
        for project in single_projects:
            project.successors = [
                np.array(p) + previous_jobs for p in project.successors
            ]
            project.successors = [
                np.hstack([p, n_jobs - 1]) for p in project.successors
            ]
            previous_jobs += project.n_jobs

        successors = np.concatenate([project.successors for project in single_projects])
        successors = (
            list([np.arange(1, n_jobs)]) + list(successors) + list([np.array([])])
        )

        predecessors = [[] for _ in successors]
        for i, succ in enumerate(successors):
            for s in succ:
                predecessors[int(s)].append(i)

        for project in single_projects:
            if project.n_resources < n_resources:
                r = n_resources - project.n_resources
                project.required_resources = np.hstack(
                    (project.required_resources, np.zeros((project.n_jobs, r)))
                )
                project.resource_availability = np.hstack(
                    (project.resource_availability, np.zeros(r))
                )
                project.n_resources = n_resources

        required_resources = np.concatenate(
            [project.required_resources for project in single_projects], axis=0
        )

        required_resources = (
            list([np.zeros(n_resources)])
            + list(required_resources)
            + list([np.zeros(n_resources)])
        )

        max_required = np.max(
            np.vstack([project.required_resources for project in single_projects]),
            axis=0,
        )
        min_availability = np.min(
            np.vstack([project.resource_availability for project in single_projects]),
            axis=0,
        )
        min_resource_availability = np.max(
            np.vstack([max_required, min_availability]), axis=0
        )

        max_resource_availability = np.sum(
            [
                project.resource_availability
                - np.max(project.required_resources, axis=0)
                for project in single_projects
            ],
            axis=0,
        )

        resource_availability = np.random.randint(
            low=min_resource_availability, high=max_resource_availability
        )

        # TODO: Add transfer_times
        transfer_times = np.zeros((n_jobs, n_jobs))

        return MultiProject(
            n_jobs,
            n_resources,
            np.array(durations),
            successors,
            predecessors,
            required_resources,
            np.array(resource_availability),
            np.array(transfer_times),
        )
