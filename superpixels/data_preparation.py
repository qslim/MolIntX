import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import itertools

import dgl
import torch
import torch.utils.data

import time

import csv
from sklearn.model_selection import StratifiedShuffleSplit

from dgl.data.utils import load_graphs, save_graphs

from tqdm import tqdm

from utils.local_encoding import build_local_encoding


def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1)) / kth
    except ValueError:  # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1] * num_nodes).reshape(num_nodes, 1)

    return sigma + 1e-8  # adding epsilon to avoid zero value of sigma


def compute_adjacency_matrix_images(coord, feat, use_feat=True, kth=8):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)

    if use_feat:
        # Compute feature distance
        f_dist = cdist(feat, feat)
        # Compute adjacency
        A = np.exp(- (c_dist / sigma(c_dist)) ** 2 - (f_dist / sigma(f_dist)) ** 2)
    else:
        A = np.exp(- (c_dist / sigma(c_dist)) ** 2)

    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A


def compute_edges_list(A, kth=8 + 1):
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth

    if num_nodes > 9:
        knns = np.argpartition(A, new_kth - 1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth - 1, axis=-1)[:, new_kth:-1]  # NEW
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A  # NEW

        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)  # NEW
            knns = knns[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)
    return knns, knn_values  # NEW


class SuperPixDGL(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 dataset,
                 split,
                 config,
                 use_mean_px=True,
                 use_coord=True):

        self.dataset = dataset
        self.data_dir = data_dir
        self.split = split

        if self.dataset == 'MNIST':
            self.pre_processed_file_path = os.path.join(data_dir, 'mnist_75sp_%s_processed' % self.split)
            self.labels_file_path = os.path.join(data_dir, 'mnist_75sp_%s_labels' % self.split)
            self.img_size = 28
            self.raw_file = 'mnist_75sp_%s.pkl'
        else:
            raise ValueError('Unknown dataset name.')

        self.graph_lists = []

        self.use_mean_px = use_mean_px
        self.use_coord = use_coord
        # self.n_samples = len(self.labels)

        self.config = config

        self._prepare()

    def _prepare(self):
        if os.path.exists(self.pre_processed_file_path):
            print("Loading the cached file for the %s set... (NOTE: delete it if you change the preprocessing settings)" % (self.split.upper()))
            self.graph_lists, _ = load_graphs(self.pre_processed_file_path)
            self.graph_labels = torch.load(self.labels_file_path)

            # assert len(self.graph_lists) == self.num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
            # assert len(self.graph_labels) == self.num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        else:
            with open(os.path.join(self.data_dir, self.raw_file % self.split), 'rb') as f:
                self.labels, self.sp_data = pickle.load(f)
                self.graph_labels = torch.LongTensor(self.labels)

            # print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))
            self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
            for index, sample in enumerate(tqdm(self.sp_data, desc="Pre-processing")):
                mean_px, coord = sample[:2]

                try:
                    coord = coord / self.img_size
                except AttributeError:
                    VOC_has_variable_image_sizes = True

                if self.use_mean_px:
                    A = compute_adjacency_matrix_images(coord, mean_px)  # using super-pixel locations + features
                else:
                    A = compute_adjacency_matrix_images(coord, mean_px, False)  # using only super-pixel locations
                edges_list, edge_values_list = compute_edges_list(A)  # NEW

                N_nodes = A.shape[0]

                mean_px = mean_px.reshape(N_nodes, -1)
                coord = coord.reshape(N_nodes, 2)
                x = np.concatenate((mean_px, coord), axis=1)

                edge_values_list = edge_values_list.reshape(-1)  # NEW # TO DOUBLE-CHECK !

                self.node_features.append(x)
                self.edge_features.append(edge_values_list)  # NEW
                self.Adj_matrices.append(A)
                self.edges_lists.append(edges_list)

            for index, _ in enumerate(tqdm(self.sp_data, desc="Pre-processing")):
                g = dgl.DGLGraph()
                g.add_nodes(self.node_features[index].shape[0])
                g.ndata['feat'] = torch.Tensor(self.node_features[index]).half().float()

                for src, dsts in enumerate(self.edges_lists[index]):
                    # handling for 1 node where the self loop would be the only edge
                    # since, VOC Superpixels has few samples (5 samples) with only 1 node
                    if self.node_features[index].shape[0] == 1:
                        g.add_edges(src, dsts)
                    else:
                        g.add_edges(src, dsts[dsts != src])

                # adding edge features for Residual Gated ConvNet
                edge_feat_dim = g.ndata['feat'].shape[1]  # dim same as node feature dim
                # g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim).half().float()
                g.edata['feat'] = torch.Tensor(self.edge_features[index]).unsqueeze(1).half().float()  # NEW

                g = dgl.remove_self_loop(g)
                g = dgl.add_self_loop(g)

                g = build_local_encoding(g, self.config)

                self.graph_lists.append(g)
            print('Saving...')
            save_graphs(self.pre_processed_file_path, self.graph_lists)
            torch.save(self.graph_labels, self.labels_file_path)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


class SuperPixDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name, num_val=5000, config=None):
        """
            Takes input standard image dataset name (MNIST/CIFAR10)
            and returns the superpixels graph.

            This class uses results from the above SuperPix class.
            which contains the steps for the generation of the Superpixels
            graph from a superpixel .pkl file that has been given by
            https://github.com/bknyaz/graph_attention_pool

            Please refer the SuperPix class for details.
        """
        t_data = time.time()
        self.name = name

        use_mean_px = True  # using super-pixel locations + features
        use_mean_px = False  # using only super-pixel locations
        if use_mean_px:
            print('Adj matrix defined from super-pixel locations + features')
        else:
            print('Adj matrix defined from super-pixel locations (only)')
        use_coord = True

        self.config = config

        self.test = SuperPixDGL("./data/superpixels", dataset=self.name, split='test',
                                config = self.config,
                                use_mean_px=use_mean_px,
                                use_coord=use_coord)

        self.train_ = SuperPixDGL("./data/superpixels", dataset=self.name, split='train',
                                  config=self.config,
                                  use_mean_px=use_mean_px,
                                  use_coord=use_coord)

        _val_graphs, _val_labels = self.train_[:num_val]
        _train_graphs, _train_labels = self.train_[num_val:]

        self.val = DGLFormDataset(_val_graphs, _val_labels)
        self.train = DGLFormDataset(_train_graphs, _train_labels)

        print("[I] Data load time: {:.4f}s".format(time.time() - t_data))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        # tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        # tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        # snorm_n = torch.cat(tab_snorm_n).sqrt()
        # tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        # tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        # snorm_e = torch.cat(tab_snorm_e).sqrt()
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels
