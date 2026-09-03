from TSP import TSPInstance

def sparse_instances(instances: list[TSPInstance]):
    for instance in instances:
        instance.compute_sparse_adj()
    return instances