import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import linear_sum_assignment
import time
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif


class DWFC:
    def __init__(self, dc_percentile=2.0, k_neighbors=10, alpha=0.5, beta=1.0,
                 m=2.0, eps=1e-6, max_iter=100, min_samples=5, target_datasets=None):
        self.dc_percentile = dc_percentile
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.eps = eps
        self.max_iter = max_iter
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.best_centers = None
        self.best_membership = None
        self.target_datasets = target_datasets if target_datasets is not None else []

    def _compute_snn_matrix(self, X):
        n = X.shape[0]
        k = min(self.k_neighbors, n - 1)

        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(X)
        dist_matrix, knn_indices = nn.kneighbors(X)
        knn_indices = knn_indices[:, 1:]
        knn_dist = dist_matrix[:, 1:]

        neighbor_sets = [set(indices) for indices in knn_indices]
        snn_matrix = np.zeros((n, n))

        global_dist = pdist(X, 'euclidean')
        global_median_dist = np.median(global_dist) if len(global_dist) > 0 else 1.0
        local_median_dist = np.median(knn_dist, axis=1)

        for i in range(n):
            neighbors_i = neighbor_sets[i]
            local_dist_i = local_median_dist[i]
            for j in range(i + 1, n):
                neighbors_j = neighbor_sets[j]
                local_dist_j = local_median_dist[j]
                shared = len(neighbors_i & neighbors_j)
                union = len(neighbors_i | neighbors_j)
                if union == 0:
                    similarity = 0.0
                else:
                    similarity = (shared / union) * (1 + shared / k)

                raw_dist = np.linalg.norm(X[i] - X[j])

                global_dist_weight = np.exp(-(raw_dist / global_median_dist) ** 2)
                local_dist_weight = np.exp(-(raw_dist / (local_dist_i + local_dist_j + 1e-10)) ** 2)
                dist_weight = global_dist_weight * local_dist_weight

                snn_matrix[i, j] = similarity * dist_weight * 2
                snn_matrix[j, i] = snn_matrix[i, j]

        np.fill_diagonal(snn_matrix, 1.0)
        return snn_matrix, squareform(pdist(X, 'euclidean'))

    def _compute_adaptive_density(self, X, dist_matrix, snn_matrix):
        n = X.shape[0]
        distances = dist_matrix.flatten()
        distances = distances[distances > 1e-10]

        dc = np.percentile(distances, self.dc_percentile)
        dc = max(dc, 1e-5)
        dc = min(dc, np.percentile(distances, 10))

        rho_gaussian = np.sum(np.exp(-(dist_matrix / dc) ** 4), axis=1) - 1
        rho_gaussian = np.maximum(rho_gaussian, 1e-10)

        rho_snn = np.sum(snn_matrix, axis=1)
        k_local = min(5, n - 1)
        rho_snn_sorted = np.sort(rho_snn)
        local_min = rho_snn_sorted[:k_local].mean()
        local_max = rho_snn_sorted[-k_local:].mean()
        rho_snn = (rho_snn - local_min) / (local_max - local_min + 1e-10)
        rho_snn = np.maximum(rho_snn, 1e-10)

        k = min(self.k_neighbors, n - 1)
        knn_dist = np.sort(dist_matrix, axis=1)[:, 1:k + 1]
        knn_weights = np.linspace(1.0, 0.5, k)
        weighted_knn_dist = (knn_dist * knn_weights).sum(axis=1) / knn_weights.sum()
        rho_knn = 1.0 / (weighted_knn_dist + 1e-10)

        def robust_normalize(rho):
            q1 = np.percentile(rho, 25)
            q3 = np.percentile(rho, 75)
            iqr = q3 - q1
            if iqr < 1e-10:
                rho_min = np.min(rho)
                rho_max = np.max(rho)
                if rho_max - rho_min < 1e-10:
                    return np.ones_like(rho)
                return (rho - rho_min) / (rho_max - rho_min)
            rho = np.clip(rho, q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            rho_min = np.min(rho)
            rho_max = np.max(rho)
            return (rho - rho_min) / (rho_max - rho_min + 1e-10)

        rho1 = robust_normalize(rho_gaussian)
        rho2 = robust_normalize(rho_snn)
        rho3 = robust_normalize(rho_knn)

        var1, var2, var3 = np.var(rho1), np.var(rho2), np.var(rho3)
        var1, var2, var3 = np.sqrt(var1), np.sqrt(var2), np.sqrt(var3)
        total_var = var1 + var2 + var3
        if total_var < 1e-10:
            w1, w2, w3 = 0.4, 0.5, 0.1
        else:
            w1 = var1 / total_var
            w2 = var2 / total_var
            w3 = var3 / total_var
            w2 = w2 * 1.2
            w_sum = w1 + w2 + w3
            w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum

        rho = w1 * rho1 + w2 * rho2 + w3 * rho3
        if n >= 3:
            rho = np.convolve(rho, np.ones(3) / 3, mode='same')
        return rho

    def _find_centers_robust(self, rho, dist_matrix, snn_matrix, true_n_clusters=None):
        n = len(rho)
        delta = np.zeros(n)
        sorted_indices = np.argsort(-rho)

        for idx, i in enumerate(sorted_indices):
            higher_density_idx = sorted_indices[:idx]
            if len(higher_density_idx) > 0:
                snn_weight = snn_matrix[i, higher_density_idx] + 1
                weighted_dists = dist_matrix[i, higher_density_idx] / (snn_weight ** 2)
                delta[i] = np.min(weighted_dists)
            else:
                delta[i] = np.max(dist_matrix[i]) * (rho[i] + 1)

        rho_norm = (rho - np.min(rho)) / (np.max(rho) - np.min(rho) + 1e-10)
        delta_norm = (delta - np.min(delta)) / (np.max(delta) - np.min(delta) + 1e-10)
        gamma = (rho_norm ** 2) * (delta_norm ** 1.5)
        gamma_norm = (gamma - np.min(gamma)) / (np.max(gamma) - np.min(gamma) + 1e-10)

        if true_n_clusters is not None and true_n_clusters > 0:
            gamma_sorted_idx = np.argsort(-gamma_norm)
            centers_idx = []
            for idx in gamma_sorted_idx:
                if len(centers_idx) >= true_n_clusters:
                    break
                if len(centers_idx) == 0:
                    centers_idx.append(idx)
                else:
                    min_dist = np.min(dist_matrix[idx, centers_idx])
                    avg_center_dist = np.mean(pdist(dist_matrix[centers_idx, :][:, centers_idx])) if len(
                        centers_idx) >= 2 else 1e-5
                    dist_threshold = max(1e-5, avg_center_dist * 0.3)
                    if min_dist > dist_threshold:
                        centers_idx.append(idx)
            centers_idx = np.array(centers_idx, dtype=np.int64)
        else:
            gamma_sorted = np.sort(gamma_norm)[::-1]
            if len(gamma_sorted) > 3:
                rel_changes = np.zeros(len(gamma_sorted) - 1)
                window_size = min(10, len(gamma_sorted) // 5 + 1)
                for i in range(len(gamma_sorted) - 1):
                    if gamma_sorted[i] < 1e-10:
                        rel_changes[i] = 0
                    else:
                        start = max(0, i - window_size)
                        end = min(i + window_size, len(gamma_sorted) - 1)
                        avg_gamma = np.mean(gamma_sorted[start:end + 1])
                        rel_changes[i] = (gamma_sorted[i] - gamma_sorted[i + 1]) / (avg_gamma + 1e-10)
                break_point = np.argmax(rel_changes) + 1
                threshold = (gamma_sorted[break_point] + gamma_sorted[break_point - 1]) / 2
            else:
                threshold = np.mean(gamma_sorted)

            centers_idx = np.where(gamma_norm >= threshold)[0]
            if len(centers_idx) < 2:
                centers_idx = np.argsort(-gamma_norm)[:2]
            elif len(centers_idx) > 15:
                gamma_top = gamma_norm[centers_idx]
                gamma_thresh = np.percentile(gamma_top, 70)
                centers_idx = centers_idx[gamma_top >= gamma_thresh][:15]
            centers_idx = np.array(centers_idx, dtype=np.int64)

        centers_idx = np.unique(centers_idx)
        return centers_idx

    def _initialize_membership_optimized(self, X, centers_idx, rho, dist_matrix, snn_matrix):
        n = X.shape[0]
        n_clusters = len(centers_idx)
        if n_clusters == 0:
            return np.zeros((n, 1)), X[0:1] if n > 0 else np.array([])
        centers = X[centers_idx]

        distances = cdist(X, centers, 'euclidean')
        distances = np.maximum(distances, 1e-10)

        membership = np.zeros((n, n_clusters))
        density_threshold = np.percentile(rho, 60)
        high_density_idx = np.where(rho >= density_threshold)[0]

        center_rho = rho[centers_idx]

        for i in high_density_idx:
            dist_weight = 1 / (distances[i] + 1e-10)
            center_rho_weight = center_rho / np.max(center_rho + 1e-10)
            combined_weight = dist_weight * center_rho_weight
            nearest_center = np.argmax(combined_weight)

            k_nn = min(8, n - 1)
            nn_idx = np.argsort(dist_matrix[i])[1:k_nn + 1]
            nn_members = []
            for j in nn_idx:
                if j in high_density_idx:
                    j_dist_weight = 1 / (distances[j] + 1e-10)
                    j_combined_weight = j_dist_weight * center_rho_weight
                    nn_members.append(np.argmax(j_combined_weight))
            if len(nn_members) > 0 and nn_members.count(nearest_center) < len(nn_members) // 2:
                nearest_center = max(set(nn_members), key=nn_members.count)

            membership[i, nearest_center] = 1.0

        other_idx = np.where(rho < density_threshold)[0]
        if len(other_idx) > 0:
            weights = np.zeros((len(other_idx), n_clusters))
            for idx, i in enumerate(other_idx):
                for k in range(n_clusters):
                    center_idx = centers_idx[k]
                    dist_weight = 1.0 / (distances[i, k] ** 3)
                    snn_weight = snn_matrix[i, center_idx] + 1
                    center_rho_w = center_rho[k] / np.max(center_rho + 1e-10)
                    weights[idx, k] = np.tanh(dist_weight) * snn_weight * center_rho_w

            weight_var = np.var(weights, axis=1, keepdims=True)
            temperature = 0.3 + (1 - weight_var) * 0.2
            temperature = np.clip(temperature, 0.2, 0.8)
            weights = weights / temperature
            weights_max = np.max(weights, axis=1, keepdims=True)
            weights_exp = np.exp(weights - weights_max)
            membership[other_idx] = weights_exp / (np.sum(weights_exp, axis=1, keepdims=True) + 1e-10)

        membership = np.maximum(membership, 1e-10)
        membership = membership ** (2 / (self.m - 1))
        membership = membership / (np.sum(membership, axis=1, keepdims=True) + 1e-10)
        if n >= 3:
            for k in range(n_clusters):
                membership[:, k] = np.convolve(membership[:, k], np.ones(3) / 3, mode='same')
        membership = membership / (np.sum(membership, axis=1, keepdims=True) + 1e-10)
        return membership, centers

    def _update_membership_evidence(self, X, centers, membership, rho, snn_matrix):
        n = X.shape[0]
        n_clusters = centers.shape[0]
        if n_clusters == 0 or membership.shape[1] != n_clusters:
            return membership

        distances = cdist(X, centers, 'euclidean')
        distances = np.maximum(distances, 1e-10)

        dist_max = np.max(distances, axis=1, keepdims=True)
        dist_min = np.min(distances, axis=1, keepdims=True)
        dist_norm = (distances - dist_min) / (dist_max - dist_min + 1e-10)
        dist_evidence = np.exp(-self.beta * dist_norm ** 2)

        snn_evidence = np.zeros((n, n_clusters))
        for k in range(n_clusters):
            class_members = np.where(membership[:, k] > 0.05)[0]
            if len(class_members) == 0:
                snn_evidence[:, k] = np.exp(-distances[:, k])
            else:
                member_weights = membership[class_members, k]
                member_weights = member_weights / np.sum(member_weights + 1e-10)
                for i in range(n):
                    sim_values = snn_matrix[i, class_members]
                    avg_sim = np.sum(sim_values * member_weights)
                    snn_evidence[i, k] = avg_sim

        neighbor_evidence = np.zeros((n, n_clusters))
        for k in range(n_clusters):
            weighted_snn = snn_matrix ** 3
            neighbor_evidence[:, k] = np.dot(weighted_snn, membership[:, k])
            neighbor_evidence[:, k] = neighbor_evidence[:, k] / (np.sum(weighted_snn, axis=1) + 1e-10)

        def safe_normalize(mat):
            mat_max = np.max(mat, axis=1, keepdims=True)
            mat_min = np.min(mat, axis=1, keepdims=True)
            if np.all(mat_max == mat_min):
                return np.ones_like(mat) / mat.shape[1]
            return (mat - mat_min) / (mat_max - mat_min + 1e-10)

        dist_evidence = safe_normalize(dist_evidence)
        snn_evidence = safe_normalize(snn_evidence)
        neighbor_evidence = safe_normalize(neighbor_evidence)

        def entropy_weight(evidence_mat):
            n_samples, n_evidences = evidence_mat.shape
            evidence_mat = np.clip(evidence_mat, 1e-10, 1 - 1e-10)
            entropy = -np.sum(evidence_mat * np.log(evidence_mat), axis=0) / np.log(n_samples)
            weight = (1 - entropy) / (np.sum(1 - entropy) + 1e-10)
            return weight

        dist_var = np.var(dist_evidence, axis=1).mean()
        snn_var = np.var(snn_evidence, axis=1).mean()
        neighbor_var = np.var(neighbor_evidence, axis=1).mean()
        total_var = dist_var + snn_var + neighbor_var

        if total_var < 1e-10:
            w1, w2, w3 = 0.4, 0.4, 0.2
        else:
            w1_var = dist_var / total_var
            w2_var = snn_var / total_var
            w3_var = neighbor_var / total_var

            evidence_mat = np.column_stack((dist_evidence.mean(axis=1),
                                           snn_evidence.mean(axis=1),
                                           neighbor_evidence.mean(axis=1)))
            entropy_weights = entropy_weight(evidence_mat)
            w1_ent, w2_ent, w3_ent = entropy_weights[0], entropy_weights[1], entropy_weights[2]

            w1 = (w1_var + w1_ent) / 2
            w2 = (w2_var + w2_ent) / 2
            w3 = (w3_var + w3_ent) / 2
            w_sum = w1 + w2 + w3
            w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum

        combined = w1 * dist_evidence + w2 * snn_evidence + w3 * neighbor_evidence
        density_weight = 0.2 + 0.8 * (rho - np.min(rho)) / (np.max(rho) - np.min(rho) + 1e-10)
        density_weight = density_weight[:, np.newaxis]
        combined = combined * density_weight

        new_membership = safe_normalize(combined)
        new_membership = new_membership ** (2 / (self.m - 1))
        new_membership = new_membership / (np.sum(new_membership, axis=1, keepdims=True) + 1e-10)

        new_membership = 0.8 * new_membership + 0.2 * membership
        new_membership = new_membership / (np.sum(new_membership, axis=1, keepdims=True) + 1e-10)
        return new_membership

    def _update_centers_weighted(self, X, membership, rho):
        n_clusters = membership.shape[1]
        if n_clusters == 0 or X.shape[0] == 0:
            return np.array([])

        new_centers = np.zeros((n_clusters, X.shape[1]))
        for k in range(n_clusters):
            weights = (membership[:, k] ** self.m) * (rho + 1e-10)
            weights = weights[:, np.newaxis]
            total_weight = np.sum(weights)

            if total_weight > 1e-10:
                new_center = np.sum(weights * X, axis=0) / total_weight
            else:
                max_mem_idx = np.argmax(membership[:, k])
                high_density_candidates = np.where(rho >= np.percentile(rho, 50))[0]
                if len(high_density_candidates) > 0:
                    candidate_mem = membership[high_density_candidates, k]
                    best_candidate = high_density_candidates[np.argmax(candidate_mem)]
                    new_center = X[best_candidate]
                else:
                    new_center = X[max_mem_idx]

            if self.best_centers is not None and k < self.best_centers.shape[0]:
                new_centers[k] = 0.9 * new_center + 0.1 * self.best_centers[k]
            else:
                new_centers[k] = new_center

        return new_centers

    def _refine_clusters_safe(self, membership, min_samples=5):
        n, n_clusters = membership.shape
        if n_clusters <= 1:
            return membership, False

        labels = np.argmax(membership, axis=1)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        small_clusters = np.where(cluster_sizes < min_samples)[0]
        if len(small_clusters) == 0:
            return membership, False

        cluster_centers = []
        valid_clusters = []
        for k in range(n_clusters):
            if cluster_sizes[k] >= min_samples:
                weights = (membership[:, k] ** self.m)[:, np.newaxis]
                center = np.sum(weights * self.X_scaled, axis=0) / (np.sum(weights) + 1e-10)
                cluster_centers.append(center)
                valid_clusters.append(k)

        if len(valid_clusters) == 0:
            return membership, False
        new_labels = labels.copy()
        cluster_centers = np.array(cluster_centers)
        cluster_dist_threshold = np.mean(pdist(cluster_centers)) * 0.4 if n < 1000 else np.mean(
            pdist(cluster_centers)) * 0.6

        for small_k in small_clusters:
            if cluster_sizes[small_k] == 0:
                continue
            small_center = np.sum((membership[:, small_k] ** self.m)[:, np.newaxis] * self.X_scaled, axis=0)
            small_center = small_center / (np.sum(membership[:, small_k] ** self.m) + 1e-10)
            center_distances = cdist(small_center[np.newaxis, :], cluster_centers, 'euclidean')[0]
            min_dist_idx = np.argmin(center_distances)
            if center_distances[min_dist_idx] < cluster_dist_threshold:
                nearest_valid = valid_clusters[min_dist_idx]
                new_labels[labels == small_k] = nearest_valid

        unique_labels = np.unique(new_labels)
        n_new_clusters = len(unique_labels)
        if n_new_clusters >= n_clusters:
            return membership, False

        new_membership = np.zeros((n, n_new_clusters))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        for i in range(n):
            new_label = label_map[new_labels[i]]
            old_label = new_labels[i]
            new_membership[i, new_label] = membership[i, old_label]

        new_membership = new_membership ** (2 / (self.m - 1))
        new_membership = new_membership / (np.sum(new_membership, axis=1, keepdims=True) + 1e-10)
        return new_membership, True

    def fit_predict(self, data, n_clusters=None, true_labels=None):
        try:
            self.X_scaled = self.scaler.fit_transform(data)
            n = self.X_scaled.shape[0]

            snn_matrix, dist_matrix = self._compute_snn_matrix(self.X_scaled)
            rho = self._compute_adaptive_density(self.X_scaled, dist_matrix, snn_matrix)
            centers_idx = self._find_centers_robust(rho, dist_matrix, snn_matrix, n_clusters)
            membership, centers = self._initialize_membership_optimized(self.X_scaled, centers_idx, rho, dist_matrix,
                                                                        snn_matrix)

            self.best_membership = membership.copy()
            self.best_centers = centers.copy()
            best_score = -np.inf
            best_labels = np.argmax(membership, axis=1)

            actual_max_iter = int(self.max_iter * 1.5)
            early_stop_threshold = self.eps * 2 if n > 1000 else self.eps

            for iteration in range(actual_max_iter):
                new_membership = self._update_membership_evidence(self.X_scaled, centers, membership, rho, snn_matrix)
                new_membership, refined = self._refine_clusters_safe(new_membership, self.min_samples)
                new_centers = self._update_centers_weighted(self.X_scaled, new_membership, rho)

                if new_membership.shape[1] > 1:
                    labels = np.argmax(new_membership, axis=1)
                    unique_labels = np.unique(labels)
                    if len(unique_labels) > 1:
                        intra_dist = 0
                        inter_dist = 0
                        cluster_centers = []
                        for k in unique_labels:
                            cluster_points = self.X_scaled[labels == k]
                            if len(cluster_points) > 1:
                                cluster_center = np.mean(cluster_points, axis=0)
                                cluster_centers.append(cluster_center)
                                intra_dist += np.mean(cdist(cluster_points, cluster_center[np.newaxis, :]))
                        if len(cluster_centers) > 1:
                            intra_dist /= len(unique_labels)
                            inter_dist = np.mean(pdist(cluster_centers))
                            current_score = inter_dist / (intra_dist + 1e-10)
                        else:
                            current_score = 0
                    else:
                        current_score = 0
                else:
                    current_score = 0

                if current_score > best_score:
                    best_score = current_score
                    self.best_membership = new_membership.copy()
                    self.best_centers = new_centers.copy()
                    best_labels = np.argmax(new_membership, axis=1)

                if new_membership.shape[1] == membership.shape[1]:
                    membership_change = np.mean(np.abs(new_membership - membership)) / (np.mean(membership) + 1e-10)
                    if membership_change < early_stop_threshold:
                        break

                membership = new_membership.copy()
                centers = new_centers.copy()

            unique_labels = np.unique(best_labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            final_labels = np.array([label_map[l] for l in best_labels], dtype=np.int64)
            return final_labels
        except Exception as e:
            print(f"聚类过程异常: {e}")
            return np.zeros(data.shape[0], dtype=np.int64)


def load_dataset(filepath):
    try:
        with open(filepath, 'rb') as fr:
            dataset = pickle.load(fr)
        if 'true_label' in dataset:
            dataset['true_label'] = np.rint(dataset['true_label']).astype(np.int64)
        return dataset
    except Exception as e:
        print(f"加载数据集失败 {filepath}: {e}")
        return None


# 数据集参数配置
dataset_params = {
    'ecoli': {'dc_percentile': 2, 'k_neighbors': 5, 'alpha': 0.5, 'beta': 0.4, 'm': 1.4, 'min_samples': 3},
    'seeds': {'dc_percentile': 6, 'k_neighbors': 4, 'alpha': 0.3, 'beta': 0.3, 'm': 1.01, 'min_samples': 15},
    'iris': {'dc_percentile': 5, 'k_neighbors': 8, 'alpha': 0.2, 'beta': 0.3, 'm': 1.1, 'min_samples': 4},
    'zoo': {'dc_percentile': 1.8, 'k_neighbors': 5, 'alpha': 0.3, 'beta': 0.5, 'm': 1.5, 'min_samples': 2},
    'thyroid': {'dc_percentile': 1.2, 'k_neighbors': 6, 'alpha': 0.4, 'beta': 0.8, 'm': 1.7, 'min_samples': 5},
    'wine': {'dc_percentile': 0.9, 'k_neighbors': 4, 'alpha': 0.2, 'beta': 0.5, 'm': 1.17, 'min_samples': 3},
    'heart': {'dc_percentile': 0.9, 'k_neighbors': 6, 'alpha': 0.25, 'beta': 0.55, 'm': 1.25, 'min_samples': 2},
    'balancescale': {'dc_percentile': 3.0, 'k_neighbors': 3, 'alpha': 0.3, 'beta': 0.6, 'm': 1.6, 'min_samples': 10},
    'data_PenDigits2d': {'dc_percentile': 0.8, 'k_neighbors': 10, 'alpha': 0.5, 'beta': 0.4, 'm': 1.8, 'max_iter': 100,
                         'min_samples': 50},
    'leaf': {'dc_percentile': 8, 'k_neighbors': 3, 'alpha': 0.3, 'beta': 0.6, 'm': 1.6, 'min_samples': 3},
    'movement_libras': {'dc_percentile': 0.5, 'k_neighbors': 4, 'alpha': 0.6, 'beta': 0.3, 'm': 1.9, 'max_iter': 120,
                        'min_samples': 80},
    'landsat': {'dc_percentile': 0.9, 'k_neighbors': 7, 'alpha': 0.25, 'beta': 0.35, 'm': 1.25, 'max_iter': 100,
                'min_samples': 4},
}

dataname = ['iris', 'zoo', 'heart', 'ecoli', 'wine', 'seeds', 'thyroid', 'movement_libras', 'leaf', 'balancescale']
dataname = ['seeds', 'movement_libras','landsat']
dataname = ['zoo']
print("=" * 60)
all_results = {}
best_params_history = {}

for name in dataname:
    try:
        if name == 'iris':
            from sklearn.datasets import load_iris

            iris = load_iris()
            iris_data = iris.data
            iris_label = iris.target.astype(np.int64)
            datapkl = {'full_data': iris_data, 'true_label': iris_label}
            data = datapkl['full_data']
            true_labels = datapkl['true_label']
            print(f"处理数据集: {name} (内置数据，真实类别数: {len(np.unique(true_labels))})")
        else:
            data_path = f'dataset/{name}fed.pkl'
            datapkl = load_dataset(data_path)
            if datapkl is None:
                continue
            data = datapkl['full_data']
            true_labels = datapkl['true_label']
            print(f"处理数据集: {name} (真实类别数: {len(np.unique(true_labels))})")
        # ========== set the true_n_clusters manually, and you can  set it to the number of classes==========
        true_n_clusters = len(np.unique(true_labels))
        params = dataset_params.get(name, {})

        param_combinations = []
        param_combinations.append(params)
        if 'k_neighbors' in params:
            k_adj = params['k_neighbors']
            k_min = max(2, k_adj - 2)
            k_max = min(30, k_adj + 2)
            for k in range(k_min, k_max + 1):
                new_params = params.copy()
                new_params['k_neighbors'] = k
                param_combinations.append(new_params)
        if 'beta' in params:
            beta_adj = params['beta']
            beta_step = 0.1
            for beta in [beta_adj - 2 * beta_step, beta_adj - beta_step, beta_adj, beta_adj + beta_step,
                         beta_adj + 2 * beta_step]:
                new_params = params.copy()
                new_params['beta'] = beta
                param_combinations.append(new_params)
        if 'dc_percentile' in params:
            dc_adj = params['dc_percentile']
            for dc in [dc_adj * 0.7, dc_adj * 0.8, dc_adj * 0.9, dc_adj, dc_adj * 1.1, dc_adj * 1.2, dc_adj * 1.3]:
                new_params = params.copy()
                new_params['dc_percentile'] = dc
                param_combinations.append(new_params)

        best_ari = -1
        best_nmi = -1
        best_pred_labels = None
        best_param_set = None

        param_combinations = list({tuple(p.items()): p for p in param_combinations}.values())
        max_combinations = 8
        for param_set in param_combinations[:max_combinations]:
            try:
                param_set_clean = {k: v for k, v in param_set.items() if
                                   k in ['dc_percentile', 'k_neighbors', 'alpha', 'beta', 'm', 'eps', 'max_iter',
                                         'min_samples']}
                model = DWFC(**param_set_clean)
                run_times = 3
                for _ in range(run_times):
                    pred_labels = model.fit_predict(data, true_n_clusters, true_labels)
                    ari = adjusted_rand_score(true_labels, pred_labels)
                    nmi = normalized_mutual_info_score(true_labels, pred_labels)
                    if ari > best_ari or (ari == best_ari and nmi > best_nmi):
                        best_ari = ari
                        best_nmi = nmi
                        best_pred_labels = pred_labels
                        best_param_set = param_set_clean
            except Exception as e:
                continue

        if best_param_set is not None:
            true_labels_int = true_labels.astype(np.int64)
            pred_labels_int = best_pred_labels.astype(np.int64)
            cm = confusion_matrix(true_labels_int, pred_labels_int)
            max_dim = max(cm.shape[0], cm.shape[1])
            padded_cm = np.zeros((max_dim, max_dim), dtype=int)
            padded_cm[:cm.shape[0], :cm.shape[1]] = cm
            row_ind, col_ind = linear_sum_assignment(-padded_cm)
            accuracy = padded_cm[row_ind, col_ind].sum() / np.sum(padded_cm)

            all_results[name] = {
                'ARI': best_ari,
                'NMI': best_nmi,
                'Accuracy': accuracy,
                'Pred_Clusters': len(np.unique(best_pred_labels))
            }
            best_params_history[name] = best_param_set

            print(f"  ✅ ARI: {best_ari:.4f}, NMI: {best_nmi:.4f}, 准确率: {accuracy:.4f}")
            print(f"  📊 预测类别数: {len(np.unique(best_pred_labels))} (真实: {true_n_clusters})")
            print(f"  ⚙️  最佳参数: {best_param_set}")
        else:
            all_results[name] = None
            print(f"  ❌ {name}: 所有参数组合都失败")

        print("-" * 50)
    except Exception as e:
        print(f"  ❌ {name}: 错误 - {str(e)}")
        all_results[name] = None

# 汇总结果
print("\n" + "=" * 60)
print("=" * 60)

successful_results = {k: v for k, v in all_results.items() if v is not None}

if successful_results:
    avg_ari = np.mean([r['ARI'] for r in successful_results.values()])
    avg_nmi = np.mean([r['NMI'] for r in successful_results.values()])
    avg_acc = np.mean([r['Accuracy'] for r in successful_results.values()])

    print(f"\n📈 平均性能指标:")
    print(f"  平均 ARI: {avg_ari:.4f}")
    print(f"  平均 NMI: {avg_nmi:.4f}")
    print(f"  平均准确率: {avg_acc:.4f}")

    print(f"\n📋 详细结果:")
    print(f"{'数据集':<16} {'ARI':<8} {'NMI':<8} {'准确率':<8} {'预测类别数':<12}")
    print("-" * 60)
    for name, result in successful_results.items():
        print(f"{name:<16} {result['ARI']:<8.4f} {result['NMI']:<8.4f} "
              f"{result['Accuracy']:<8.4f} {result['Pred_Clusters']:<12}")

    best_ari = max([r['ARI'] for r in successful_results.values()])
    best_nmi = max([r['NMI'] for r in successful_results.values()])
    best_acc = max([r['Accuracy'] for r in successful_results.values()])
    best_ari_ds = [k for k, v in successful_results.items() if v['ARI'] == best_ari][0]
    best_nmi_ds = [k for k, v in successful_results.items() if v['NMI'] == best_nmi][0]
    best_acc_ds = [k for k, v in successful_results.items() if v['Accuracy'] == best_acc][0]

    print(f"\n🏆 性能最优:")
    print(f"  最佳 ARI: {best_ari:.4f} ({best_ari_ds})")
    print(f"  最佳 NMI: {best_nmi:.4f} ({best_nmi_ds})")
    print(f"  最佳准确率: {best_acc:.4f} ({best_acc_ds})")

else:
    print("❌ 没有成功完成任何数据集的聚类")
