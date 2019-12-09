import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString


def compute_score(submission_file_path, gt_file_path, complete_result_required=False):
    try:
        with open(submission_file_path, "rb") as f:
            submission = pickle.load(f)
        with open(gt_file_path, "rb") as f:
            gt = pickle.load(f)
    except Exception as e:
        raise RuntimeError("Something went wrong during loading the pickle files.")
    results = {}
    for sample_id in gt.keys():
        if sample_id not in submission:
            if complete_result_required:
                raise RuntimeError(f"Prediction missing for object {sample_id}")
            else:
                continue
        gt_sample = gt[sample_id]
        submission_sample = submission[sample_id]
        try:
            gt_adjacency, gt_coordinates = gt_sample['adjacency'], np.asarray(gt_sample['coordinates'])
            submission_adjacency, submission_coordinates = submission_sample['adjacency'], np.asarray(submission_sample['coordinates'])
            gt_graph = build_skeleton_graph(gt_adjacency)
            pred_graph = build_skeleton_graph(submission_adjacency)
            score = evaluate_skeleton(gt_graph, pred_graph, gt_coordinates, submission_coordinates, num_samples=100)
            results[sample_id] = score
        except Exception as e:
            raise RuntimeError("A problem occured while processing the submission.")
    scores = [score for score in results.values()]
    if len(scores) > 0:
        mean_score = np.mean(scores)
    else:
        mean_score = 0.0
    return results, mean_score


def load_skeleton(pkl_path):
    """
    Load a skeleton pickle file.
    The skeleton is defined by a node adjacency list and node coordinates.
    :param pkl_path: pickle path
    :return: adjacency list, coordinates array
    """
    with open(pkl_path, 'rb') as f:
        gt = pickle.load(f)
    adjacency = gt['adjacency']
    coordinates = np.asarray(gt['coordinates'])
    return adjacency, coordinates


def build_skeleton_graph(adjacency):
    """
    Build a networkx graph from an adjacency list.
    :param adjacency: adjacency list
    :return: networkx graph
    """
    graph = nx.Graph()
    for c, adj in enumerate(adjacency):
        graph.add_edges_from([(c, x) for x in adj])
    return graph


def is_valid_skeleton(graph):
    """
    Check if skeleton graph has a valid structure, i.e. is an undirected tree.
    :param graph: skeleton networkx graph
    :return: true if skeleton is valid
    """
    return nx.is_tree(graph) and not nx.is_directed(graph)


def plot_skeleton(plot_path, graph, coordinates, show_branches=False):
    """
    Plot skeleton. If show_branches then detect and colorize the branches of the skeleton, requires a valid skeleton.
    :param plot_path: plot path
    :param graph: skeleton graph
    :param coordinates: skeleton coordinates
    :param show_branches: true if detect and colorize branches
    :return:
    """
    fig = plt.figure(figsize=(2.56, 2.56))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')

    if show_branches:
        branches = get_branches(graph)
        for b in branches.values():
            plt.plot(*coordinates[b].T)
    else:
        nx.draw_networkx_edges(graph, coordinates, width=1)

    plt.savefig(plot_path)
    plt.close()


def get_branches(graph):
    """
    Detect and return branches of skeleton graph.
    :param graph: skeleton graph
    :return: dict with keys: tuples of branching nodes, values: lists of path of nodes in between
    """
    assert is_valid_skeleton(graph)
    contracted_graph = __contract_paths(graph)
    branches = {}
    for node in contracted_graph.nodes:
        for adj_node in contracted_graph[node]:
            if node < adj_node:
                branches[node, adj_node] = nx.shortest_path(graph, node, adj_node)
    return branches


def evaluate_skeleton(gt_graph, pred_graph, gt_coordinates, pred_coordinates, num_samples=100, show_plot=False,
                      plot_path=None):
    """
    Evaluate a predicted skeleton against a ground truth skeleton.
    The evaluation metric is a symmetric point registration between the skeletons.
    The score is based on the average distance between sampled points on the skeleton.
    :param gt_graph: ground truth graph
    :param pred_graph: prediction graph
    :param gt_coordinates: ground truth coordinates
    :param pred_coordinates: prediction coordinates
    :param num_samples: number of sampled points, many points -> high score accuracy but slow
    :param show_plot: true if plot the point registration
    :param plot_path: plot path if show_plot
    :return: the evaluation score
    """
    gt_points = __sample_points(gt_graph, gt_coordinates, num_samples)
    pred_points = __sample_points(pred_graph, pred_coordinates, num_samples)
    pred_to_gt_registration, pred_to_gt_distances = __match_points(pred_points, gt_points)
    gt_to_pred_registration, gt_to_pred_distances = __match_points(gt_points, pred_points)

    if show_plot:
        # assert plot_path is not None
        fig = plt.figure(figsize=(10.24, 10.24))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('equal')
        nx.draw_networkx_edges(pred_graph, pred_coordinates, width=2, edge_color='red')
        nx.draw_networkx_edges(gt_graph, gt_coordinates, width=2, edge_color='black')

        for reg in pred_to_gt_registration:
            reg_line = np.asarray(reg)
            plt.plot(*reg_line.T, linewidth=1.5, c='lightblue', zorder=0)
        for reg in gt_to_pred_registration:
            reg_line = np.asarray(reg)
            plt.plot(*reg_line.T, linewidth=1.5, c='lightgreen', zorder=0)

        # plt.savefig(plot_path)
        plt.show()
        plt.close()

    total_distances = (np.asarray(pred_to_gt_distances) + np.asarray(gt_to_pred_distances)).sum()
    avg_distance = total_distances / (2 * num_samples)
    worst_distance = 0.1
    score = max(0.0, 100.0 - (100.0 / worst_distance) * avg_distance)
    return score


def __contract_paths(graph):
    contracted_graph = graph.copy()
    for node in list(contracted_graph.nodes()):
        if contracted_graph.degree(node) == 2:
            edges = list(contracted_graph.edges(node))
            contracted_graph.add_edge(edges[0][1], edges[1][1])
            contracted_graph.remove_node(node)
    return contracted_graph


def __match_points(from_points, to_points):
    def coord(point):
        return point.x, point.y

    order = np.arange(len(from_points))
    order = sorted(order, key=lambda x: -len(from_points[x]))
    registration = []
    distances = []
    all_to_points = [x for y in to_points for x in y]
    for c in order:
        for p in from_points[c]:
            dists = [p.distance(x) for x in all_to_points]
            argmin_dist = np.argmin(dists)
            registration.append((coord(p), coord(all_to_points[argmin_dist])))
            distances.append(dists[argmin_dist])
    return registration, distances


def __sample_points(graph, coordinates, num_samples=100):
    branches = get_branches(graph)
    branch_lines = MultiLineString([coordinates[x] for x in branches.values()])

    lengths = np.asarray([x.length for x in branch_lines])
    weights = lengths / lengths.sum()
    nums = (weights * num_samples).astype(int)
    choices = np.random.choice(np.arange(len(branch_lines)), size=num_samples - nums.sum(), p=weights)
    points_per_branch = nums + np.bincount(choices, minlength=len(branch_lines))
    assert points_per_branch.sum() == num_samples
    sampled_points = []
    for line, num_pts in zip(branch_lines, points_per_branch):
        if num_pts > 1:
            sp = [line.interpolate(i / float(num_pts - 1), normalized=True) for i in range(num_pts)]
        elif num_pts == 1:
            sp = [line.interpolate(0.5)]
        else:
            sp = []
        sampled_points.append(sp)
    return sampled_points
