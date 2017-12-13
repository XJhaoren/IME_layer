#coding:utf-8 允许中文注释
import sys
import os
import numpy as np
from IME_layer import compute_IME, use_original_feature

def get_nn(x, data, k=None):
    """
    Find the k top indices and distances of index data vectors from query vector x.

    :param ndarray x:
        the query vector
    :param ndarray data:
        the index vectors
    :param int k:
        optional k to truncate return

    :returns ndarray idx:
        the indices of index vectors in ascending order of distance
    :returns ndarray dists:
        the squared distances
    """
    if k is None:
        k = len(data)

    dists = ((x - data)**2).sum(axis=1)
    idx = np.argsort(dists) 
    dists = dists[idx]

    return idx[:k], dists[:k]

def load_features(feature_dir, verbose=True):
    if type(feature_dir) == str:
        feature_dir = [feature_dir]

    for directory in feature_dir:
        for i, f in enumerate(os.listdir(directory)):
            name = os.path.splitext(f)[0]

            # Print progress
            if verbose and not i % 100:
                sys.stdout.write('\rProcessing file %i' % i)
                sys.stdout.flush()

            X = np.load(os.path.join(directory, f))

            yield X, name

    sys.stdout.write('\n')
    sys.stdout.flush()

def load_and_aggregate_features_multiprocess(feature_dir=None, agg_fn=None, region_scale_num=3):
    # 加载regional的特征和图片名
    print 'Loading reginal features %s ...' % str(feature_dir)
    features = []
    names = []
    i = 0
    for X, name in load_features(feature_dir):
        i = i + 1
        print 'Loading dataset %d' % i
        X = agg_fn(X=X, region_scale_num=region_scale_num)
        names.extend([name]*len(X))
        features.extend(X)
    features = np.reshape(features, (len(features), -1))
    names=[names]

    return features, names

def get_ap(inds, dists, query_name, index_names, groundtruth_dir, ranked_dir=None):
    """
    Given a query, index data, and path to groundtruth data, perform the query,
    and evaluate average precision for the results by calling to the compute_ap
    script. Optionally save ranked results in a file.

    :param ndarray inds:
        the indices of index vectors in ascending order of distance
    :param ndarray dists:
        the squared distances
    :param str query_name:
        the name of the query
    :param list index_names:
        the name of index items
    :param str groundtruth_dir:
        directory of groundtruth files
    :param str ranked_dir:
        optional path to a directory to save ranked list for query

    :returns float:
        the average precision for this query
    """
    if ranked_dir is not None:
        # Create dir for ranked results if needed
        if not os.path.exists(ranked_dir):
            os.makedirs(ranked_dir)
        rank_file = os.path.join(ranked_dir, '%s.txt' % query_name)
        f = open(rank_file, 'w')


    f.writelines([index_names[0][i] +'\r'+'\n'  for i in inds])
    f.close()

    groundtruth_prefix = os.path.join(groundtruth_dir, query_name)
    cmd = './compute_ap %s %s' % (groundtruth_prefix, rank_file)
    ap = os.popen(cmd).read()

    # Delete temp file
    if ranked_dir is None:
        os.remove(rank_file)

    return float(ap.strip())

def run_eval(queries_dir, groundtruth_dir, index_features, IME_params, out_dir, agg_fn):
    data, image_names = load_and_aggregate_features_multiprocess(index_features, agg_fn)
    data, _ = compute_IME(np.vstack(data), params=IME_params)

    # Iterate queries, process them, rank results, and evaluate mAP
    aps = []
    for Q, query_name in load_features(queries_dir):
        Q = agg_fn(Q)
        Q, _ = compute_IME(Q, params=IME_params)

        inds, dists = get_nn(Q, data)

        ap = get_ap(inds, dists, query_name, image_names, groundtruth_dir, out_dir)
        aps.append(ap)
        print('mAP: %f ' % ap)
        print('%s' % query_name)
        print ('\n')

    return np.array(aps).mean()

def learn_IME(features, agg_fn):
    data, _ = load_and_aggregate_features_multiprocess(features, agg_fn, region_scale_num=3)
    print "Learning IME layer....."
    _, IME_params = compute_IME(features=data)
    return IME_params


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--index_features', dest='index_features', type=str,
                        default='../feature/oxford_dataset_region_manifold',
                        help='directory containing raw features to index')
    parser.add_argument('--learn_features', dest='learn_features', type=str,
                        default='../feature/oxford_dataset_region_manifold',
                        help='directory containing raw features to learn IME')
    parser.add_argument('--queries', dest='queries', type=str,
                        default='../feature/oxford_query_region_manifold/',
                        help='directory containing image files')
    parser.add_argument('--groundtruth', dest='groundtruth', type=str,
                        default='../gt/oxford_gt/',
                        help='directory containing groundtruth files')
    parser.add_argument('--out', dest='out', type=str, default='./result/IME/oxford',
                            help='optional path to save ranked output')
    args = parser.parse_args()


    # learn IME layer
    IME_params = learn_IME(args.learn_features, use_original_feature)
    # Compute transformed features and run the evaluation
    mAP = run_eval(args.queries, args.groundtruth, args.index_features, IME_params, args.out, use_original_feature)
    print 'mAP: %f' % mAP

    exit(0)
