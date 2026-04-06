import os
import torch
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import json
from src.task2 import logger
from src.task2.entity.config import CrossVizConfig


class CrossDatasetRetrival:
    def __init__(self,config:CrossVizConfig):
        self.config=config
        with open('annotations.json', 'r') as f:
            master_annotations = json.load(f)
        

        self.label_1=input("Provide Dataset 1 Label name (e.g: Liver chunk 0):")
        embed1= input("Dense Embeddings 1| What is the name of file? (e.g., dense.pt): ")
        logger.info(f"{embed1}.... Loaded")
        emfile1=os.path.join(self.config.dense_embd,embed1)
        self.emb1 = torch.load(emfile1).numpy()
        localfile1= input("Local File Input 1| What is the name of file? (e.g., liver.npy): ")
        logger.info(f"{localfile1}.... Loaded")
        lf1=os.path.join(self.config.local_data,localfile1)
        self.raw1 = np.load(lf1)
        box_1=input('Enter Dataset 1 annotation name (e.g., ls0, ls1, hs1, hs0, ps1): ')
        self.mito_boxes_ds1 = master_annotations[box_1]

        self.label_2=input("Provide Dataset 2 label name|(e.g: pancreas chunk 0):")
        embed2= input("Dense Embeddings 2| What is the name of file? (e.g., dense.pt): ")
        logger.info(f"{embed2}.... Loaded")
        emfile2=os.path.join(self.config.dense_embd,embed2)
        self.emb2 = torch.load(emfile2).numpy()
        localfile2= input("Local File Input 2| What is the name of file? (e.g., liver.npy): ")
        logger.info(f"{localfile2}.... Loaded")
        lf2=os.path.join(self.config.local_data,localfile2)
        self.raw2 = np.load(lf2)
        box_2=input('Enter Dataset 2 annotation name (e.g., ls0, ls1, hs1, hs0, ps1): ')
        self.mito_boxes_ds2 = master_annotations[box_2]

        self.c1, self.z1, self.h1, self.w1 = self.emb1.shape
        self.c2, self.z2, self.h2, self.w2 = self.emb2.shape

        self.flat1 = normalize(self.emb1.reshape(self.c1, -1).T, norm='l2')
        self.flat2 = normalize(self.emb2.reshape(self.c2, -1).T, norm='l2')

        logger.info(
            f'[{self.label_1}] embeddings: {self.emb1.shape} | voxels: {self.flat1.shape}\n'
            f'[{self.label_2}] embeddings: {self.emb2.shape} | voxels: {self.flat2.shape}\n'
        )
    
    def mMD(self):
        n_samples= int(input("Input Number of samples(e.g 5000,10000):"))
        idx1 = np.random.choice(len(self.flat1), n_samples, replace=False)
        idx2 = np.random.choice(len(self.flat2), n_samples, replace=False)
        X = self.flat1[idx1]
        Y = self.flat2[idx2]
        dists = np.sum((X[:100, None] - Y[None, :100]) ** 2, axis=-1)
        sigma = np.median(dists) + 1e-8
        def rbf(A, B):
            d = np.sum((A[:, None] - B[None]) ** 2, axis=-1)
            return np.exp(-d / (2 * sigma))
        mmd=float(rbf(X, X).mean() + rbf(Y, Y).mean() - 2 * rbf(X, Y).mean())
        logger.info(f'MMD: {mmd}')

    def _align(self, query_vec, gallery):
        mu1  = self.flat1.mean(axis=0, keepdims=True)
        std1 = self.flat1.std(axis=0,  keepdims=True) + 1e-8
        mu2  = self.flat2.mean(axis=0, keepdims=True)
        std2 = self.flat2.std(axis=0,  keepdims=True) + 1e-8

        q_aligned = normalize((query_vec - mu1) / std1, norm='l2')
        g_aligned = normalize((gallery   - mu2) / std2, norm='l2')

        return q_aligned, g_aligned

    def _get_query_vector(self,dataset, z, y_min, y_max, x_min, x_max):
        emb = self.emb1 if dataset == 1 else self.emb2
        roi = emb[:, z, y_min:y_max, x_min:x_max]
        vec = roi.mean(axis=(1, 2))
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.reshape(1, -1)

    def _compute_cross_heatmap(self,query_vec, use_alignment=True):
        if use_alignment:
            q, gallery = self._align(query_vec, self.flat2)
        else:
            q, gallery = normalize(query_vec, norm='l2'), self.flat2

        sim = (gallery @ q.T).squeeze()
        return sim.reshape(self.z2, self.h2, self.w2)

    def _project_pca(self,vecs_list, use_alignment):
        if use_alignment:
            mu1  = self.flat1.mean(0); std1 = self.flat1.std(0) + 1e-8
            mu2  = self.flat2.mean(0); std2 = self.flat2.std(0) + 1e-8
            aligned = []
            for i, v in enumerate(vecs_list):
                mu  = mu1  if i % 2 == 0 else mu2
                std = std1 if i % 2 == 0 else std2
                aligned.append(normalize((v - mu) / std, norm='l2'))
            vecs_list = aligned

        all_vecs = np.vstack(vecs_list)
        pca      = PCA(n_components=2)
        proj     = pca.fit_transform(all_vecs)
        var      = pca.explained_variance_ratio_

        sizes   = [len(v) for v in vecs_list]
        splits  = np.cumsum(sizes[:-1])
        parts   = np.split(proj, splits)
        return parts, var
    
    def plot_embedding_space(self,
        n_background=1500,
        use_alignment=True):

        idx1 = np.random.choice(len(self.flat1), n_background, replace=False)
        idx2 = np.random.choice(len(self.flat2), n_background, replace=False)
        bg1  = self.flat1[idx1]
        bg2  = self.flat2[idx2]

        mito1 = np.vstack([
            self._get_query_vector(1, b['z'], b['y_min'], b['y_max'],
                                    b['x_min'], b['x_max'])
            for b in self.mito_boxes_ds1
        ])
        mito2 = np.vstack([
            self._get_query_vector(2, b['z'], b['y_min'], b['y_max'],
                                    b['x_min'], b['x_max'])
            for b in self.mito_boxes_ds2
        ])

        (p_bg1, p_bg2, p_m1, p_m2), var = self._project_pca(
            [bg1, bg2, mito1, mito2], use_alignment
        )

        idx1r = np.random.choice(len(self.flat1), n_background, replace=False)
        idx2r = np.random.choice(len(self.flat2), n_background, replace=False)
        raw_proj = PCA(n_components=2).fit_transform(
            np.vstack([self.flat1[idx1r], self.flat2[idx2r]])
        )
        rp_bg1 = raw_proj[:n_background]
        rp_bg2 = raw_proj[n_background:]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('#0e1410')

        for ax_idx, ax in enumerate(axes):
            ax.set_facecolor('#0e1410')
            if ax_idx == 0:
                ax.scatter(rp_bg1[:,0], rp_bg1[:,1],
                            c='#185FA5', alpha=0.25, s=4, label=self.label_1)
                ax.scatter(rp_bg2[:,0], rp_bg2[:,1],
                            c='#D85A30', alpha=0.25, s=4, label=self.label_2)
                ax.set_title('PCA — raw embeddings',
                                color='#e8f5ec', fontsize=11, pad=8)
                ax.set_xlabel('PC1', color='#6b8f72', fontsize=9)
                ax.set_ylabel('PC2', color='#6b8f72', fontsize=9)
            else:
                ax.scatter(p_bg1[:,0], p_bg1[:,1],
                            c='#185FA5', alpha=0.25, s=4, label=self.label_1)
                ax.scatter(p_bg2[:,0], p_bg2[:,1],
                            c='#D85A30', alpha=0.25, s=4, label=self.label_2)
                ax.scatter(p_m1[:,0],  p_m1[:,1],
                            c='#4ade80', s=80, marker='*', zorder=5,
                            label=f'{self.label_1} mito')
                ax.scatter(p_m2[:,0],  p_m2[:,1],
                            c='#f472b6', s=80, marker='*', zorder=5,
                            label=f'{self.label_2} mito')
                for m1v, m2v in zip(p_m1, p_m2):
                    ax.plot([m1v[0], m2v[0]], [m1v[1], m2v[1]],
                            color='white', alpha=0.2, linewidth=0.8,
                            linestyle='--')
                ax.set_title('PCA — after Z-score alignment',
                                color='#e8f5ec', fontsize=11, pad=8)
                ax.set_xlabel(f'PC1 ({var[0]*100:.1f}% var)',
                                color='#6b8f72', fontsize=9)
                ax.set_ylabel(f'PC2 ({var[1]*100:.1f}% var)',
                                color='#6b8f72', fontsize=9)

            ax.tick_params(colors='#6b8f72', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#1D9E75')
            ax.legend(fontsize=8, facecolor='#141c16',
                        edgecolor='#1D9E75', labelcolor='#e8f5ec')

        fig.suptitle(
            f'Cross-dataset embedding space — {self.label_1} vs {self.label_2}',
            color='#e8f5ec', fontsize=13, fontweight='bold',
        )
        save_name=input("Enter file save name(e.g Liver.png): ")
        save_path=os.path.join(self.config.out,"crossretrive",save_name)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info(f'Saved → {save_path}')
        plt.show()

    def mutual_nn_rate(self, use_alignment=True):
        vecs1 = np.vstack([
            self._get_query_vector(1, b['z'], b['y_min'], b['y_max'],
                                b['x_min'], b['x_max'])
            for b in self.mito_boxes_ds1
        ])
        vecs2 = np.vstack([
            self._get_query_vector(2, b['z'], b['y_min'], b['y_max'],
                                b['x_min'], b['x_max'])
            for b in self.mito_boxes_ds2
        ])

        if use_alignment:
            mu1  = self.flat1.mean(0); std1 = self.flat1.std(0) + 1e-8
            mu2  = self.flat2.mean(0); std2 = self.flat2.std(0) + 1e-8
            vecs1 = normalize((vecs1 - mu1) / std1, norm='l2')
            vecs2 = normalize((vecs2 - mu2) / std2, norm='l2')

        sim_12 = vecs1 @ vecs2.T
        nn_12  = sim_12.argmax(axis=1)

        sim_21 = vecs2 @ vecs1.T
        nn_21  = sim_21.argmax(axis=1)

        matches = []
        for i, j in enumerate(nn_12):
            matches.append({
                'ds1_id'  : self.mito_boxes_ds1[i]['id'],
                'ds2_id'  : self.mito_boxes_ds2[j]['id'],
                'sim'     : float(sim_12[i, j]),
                'mutual'  : bool(nn_21[j] == i),
            })

        mnn_rate = sum(m['mutual'] for m in matches) / len(matches)

        logger.info(f'\n[MNN] Mutual nearest-neighbour rate : {mnn_rate:.3f}')
        logger.info(f'{"DS1 ID":<10} {"DS2 ID":<10} {"Sim":>8} {"MNN":>6}')
        logger.info('─' * 36)
        for m in matches:
            logger.info(f'{m["ds1_id"]:<10} {m["ds2_id"]:<10} '
                f'{m["sim"]:>8.4f} {"YES" if m["mutual"] else "—":>6}')

        return {'mnn_rate': mnn_rate, 'matches': matches}


    def plot_cross_retrieval(self,
        use_alignment=True,):
        threshold=float(input("Input threshold to only highlight the Top '%' of matches:"))
        idx=input(f"Input Region of interest ID for BOXannotations (0, 1, or 2): ")
        query_box=self.mito_boxes_ds1[int(idx)]
        query_vec = self._get_query_vector(
            1,
            query_box['z'],
            query_box['y_min'], query_box['y_max'],
            query_box['x_min'], query_box['x_max'],
        )
        z_slice_2=query_box['z']
        sim_map = self._compute_cross_heatmap(query_vec, use_alignment)
        heatmap = sim_map[z_slice_2]
        # threshold = np.percentile(heatmap, 98.5)
        masked  = np.ma.masked_where(heatmap < threshold, heatmap)

        fig = plt.figure(figsize=(18, 6))
        fig.patch.set_facecolor('#0e1410')
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.1,
                                left=0.03, right=0.97)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(self.raw1[query_box['z']], cmap='gray')
        ax0.add_patch(mpatches.Rectangle(
            (query_box['x_min'], query_box['y_min']),
            query_box['x_max'] - query_box['x_min'],
            query_box['y_max'] - query_box['y_min'],
            linewidth=2, edgecolor='#4ade80', facecolor='none',
        ))
        ax0.set_title(f'Query [{self.label_1}]  z={query_box["z"]}',
                        color='#4ade80', fontsize=10, pad=6)

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(self.raw2[z_slice_2], cmap='gray')
        ax1.set_title(f'Target [{self.label_2}]  z={z_slice_2}',
                        color='#e8f5ec', fontsize=10, pad=6)

        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(self.raw2[z_slice_2], cmap='gray')
        im2 = ax2.imshow(masked, cmap='hot', alpha=0.65,
                            vmin=threshold, vmax=1.0)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                        label='Cosine similarity')
        ax2.set_title(f'Cross-dataset overlay  (threshold={threshold})',
                        color='#e8f5ec', fontsize=10, pad=6)

        ax3 = fig.add_subplot(gs[3])
        ax3.set_facecolor('#141c16')
        ax3.hist(heatmap.flatten(), bins=60,
                    color='#D85A30', edgecolor='#0e1410', linewidth=0.4)
        ax3.axvline(threshold, color='#4ade80', linewidth=1.5,
                    label=f'threshold={threshold}')
        ax3.axvline(heatmap.mean(), color='#f472b6', linewidth=1.2,
                    linestyle='--', label=f'mean={heatmap.mean():.3f}')
        ax3.set_xlabel('Cosine similarity', color='#6b8f72', fontsize=9)
        ax3.set_ylabel('Pixel count',       color='#6b8f72', fontsize=9)
        ax3.tick_params(colors='#6b8f72', labelsize=8)
        for sp in ax3.spines.values():
            sp.set_edgecolor('#1D9E75')
        ax3.legend(fontsize=8, labelcolor='#e8f5ec',
                    facecolor='#0e1410', edgecolor='#1D9E75')

        for ax in [ax0, ax1, ax2]:
            ax.axis('off')

        fig.suptitle(
            f'Cross-dataset retrieval — {self.label_1} → {self.label_2}',
            color='#e8f5ec', fontsize=13, fontweight='bold', y=1.02,
        )
        save_name=input("Enter file save name(e.g Liver.png): ")
        save_path=os.path.join(self.config.out,"crossretrive",save_name)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info(f'Saved → {save_path}')
        plt.show()

                    