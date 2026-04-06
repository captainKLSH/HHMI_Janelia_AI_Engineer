import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from src.task2.utils.common import save_json
import matplotlib.gridspec as gridspec
import os, gc
from src.task2.entity.config import VizConfig
from src.task2 import logger

class WithinDatasetRetrival:
    def __init__(self,config:VizConfig ):
        self.config=config
        embed= input("Dense Embeddings| What is the name of file? (e.g., dense.pt): ")
        emfile=os.path.join(self.config.dense_embd,embed)
        self.embeddings = torch.load(emfile).numpy()
        localfile= input("Local File Input| What is the name of file? (e.g., liver.npy): ")
        lf=os.path.join(self.config.local_data,localfile)
        self.lf=lf
        self.raw_data   = np.load(lf)

        self.c, self.z, self.h, self.w = self.embeddings.shape
        flat = self.embeddings.reshape(self.c, -1).T
        self.flat_embeddings = normalize(flat, norm='l2')
        logger.info(
            f'Loaded embeddings : {self.embeddings.shape}\n'
            f'Raw volume        : {self.raw_data.shape}\n'
            f'Flat index        : {self.flat_embeddings.shape}'
        )
    def get_query_vector(self, z:int, y:int, x:int, window: int=8)-> np.ndarray:
        y_end = min(y + window, self.h)
        x_end = min(x + window, self.w)
        query_patch = self.embeddings[:, z, y:y_end, x:x_end]    
        return np.mean(query_patch, axis=(1, 2)).reshape(1, -1)
    
    def get_query_from_box(self) -> np.ndarray:
        """
        Mean-pool embeddings inside a full bounding box.
        Prefer this over get_query_vector for whole-mitochondrion queries.
        """
        z=self.config.z_cord
        y_min=self.config.y_cord[0]
        y_max=self.config.y_cord[1]
        x_min=self.config.x_cord[0]
        x_max=self.config.x_cord[1]
        roi = self.embeddings[:, z, y_min:y_max, x_min:x_max]  # [C, H_roi, W_roi]
        vec = roi.mean(axis=(1, 2))
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.reshape(1, -1)

    
    def compute_heatmap(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Dot product of pre-normalised embeddings = cosine similarity.
        Returns similarity volume [Z, H, W] in [-1, 1].
        """
        q   = normalize(query_vec, norm='l2')               # [1, C]
        sim = self.flat_embeddings @ q.T                    # [N, 1]
        return sim.reshape(self.z, self.h, self.w)
    
    def precision_at_k(self,sim_map:np.ndarray,) -> dict:
        """
        Treat pixels inside the query bounding box as positives.
        Rank all pixels by similarity and compute Precision@k and Recall@k.
        """
        z=self.config.z_cord
        y_min=self.config.y_cord[0]
        y_max=self.config.y_cord[1]
        x_min=self.config.x_cord[0]
        x_max=self.config.x_cord[1]
        k=self.config.k
        # Ground-truth positive mask
        gt_mask        = np.zeros((self.z, self.h, self.w), dtype=bool)
        gt_mask[z, y_min:y_max, x_min:x_max] = True
        gt_flat        = gt_mask.flatten()
        sim_flat       = sim_map.flatten()

        # Rank pixels descending by similarity
        ranked_idx     = np.argsort(sim_flat)[::-1]
        top_k_idx      = ranked_idx[:k]

        tp             = gt_flat[top_k_idx].sum()
        total_pos      = gt_flat.sum()

        precision_k    = tp / k
        recall_k       = tp / (total_pos + 1e-8)
        f1             = (2 * precision_k * recall_k / (precision_k + recall_k + 1e-8))

        results = {
            'Dataset Name': self.lf,
            'precision@k' : round(precision_k, 4),
            'recall@k'    : round(recall_k,    4),
            'f1@k'        : round(f1,           4),
            'k'           : k,
            'total_pos'   : int(total_pos),
        }
        save_json(path=self.config.outwithin, data=results)

        logger.info('\n[Quantitative — Within-dataset]')
        for key, val in results.items():
            logger.info(f'  {key:<14}: {val}')

        return results
    
    def plot_overlay(self,sim_map:np.ndarray,
                     title:str = 'Within-dataset retrieval'):
        """
        Four-panel figure:
          [0] Raw EM slice
          [1] Full similarity heatmap
          [2] Thresholded overlay
          [3] Similarity score histogram
        """
        threshold=self.config.th
        z_slice=self.config.z_cord
        query_box =self.config.query_box
        heatmap        = sim_map[z_slice]
        masked_heatmap = np.ma.masked_where(heatmap < threshold, heatmap)

        fig = plt.figure(figsize=(18, 5))
        fig.patch.set_facecolor('#0e1410')
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.12,
                                left=0.03, right=0.97)

        panel_titles = [
            'Raw EM slice',
            'Full similarity map',
            f'Threshold ≥ {threshold}',
            'Score distribution',
        ]
        z=self.config.z_cord
        y_min=self.config.y_cord[0]
        y_max=self.config.y_cord[1]
        x_min=self.config.x_cord[0]
        x_max=self.config.x_cord[1]

        # ── Panel 0: raw slice ─────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(self.raw_data[z_slice], cmap='gray')
        if query_box == True:
            import matplotlib.patches as mpatches
            rect = mpatches.Rectangle(
                (x_min, y_min),
                (x_max - x_min),
                (y_max - y_min),
                linewidth=2, edgecolor='#4ade80', facecolor='none',
            )
            ax0.add_patch(rect)
            ax0.text(
                x_min, y_min - 6,
                'Query', color='#4ade80', fontsize=8,
            )
        

        # ── Panel 1: full heatmap ──────────────────────────────────────────────
        ax1  = fig.add_subplot(gs[1])
        im1  = ax1.imshow(heatmap, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04,
                     label='Cosine similarity')

        # ── Panel 2: thresholded overlay ───────────────────────────────────────
        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(self.raw_data[z_slice], cmap='gray')
        im2 = ax2.imshow(masked_heatmap, cmap='viridis',
                         alpha=0.65, vmin=threshold, vmax=1.0)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                     label='Cosine similarity')

        # ── Panel 3: histogram ─────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[3])
        ax3.set_facecolor('#141c16')
        ax3.hist(heatmap.flatten(), bins=60,
                 color='#1D9E75', edgecolor='#0e1410', linewidth=0.4)
        ax3.axvline(threshold, color='#f472b6', linewidth=1.5,
                    label=f'threshold={threshold}')
        ax3.axvline(heatmap.mean(), color='#4ade80', linewidth=1.2,
                    linestyle='--', label=f'mean={heatmap.mean():.3f}')
        ax3.set_xlabel('Cosine similarity', color='#6b8f72', fontsize=9)
        ax3.set_ylabel('Pixel count',       color='#6b8f72', fontsize=9)
        ax3.tick_params(colors='#6b8f72', labelsize=8)
        ax3.legend(fontsize=8, labelcolor='#e8f5ec',
                   facecolor='#0e1410', edgecolor='#1D9E75')
        for sp in ax3.spines.values():
            sp.set_edgecolor('#1D9E75')

        # ── Shared formatting ──────────────────────────────────────────────────
        for ax, ptitle in zip([ax0, ax1, ax2, ax3], panel_titles):
            ax.set_title(ptitle, color='#e8f5ec', fontsize=10, pad=6)
            if ax != ax3:
                ax.axis('off')

        fig.suptitle(f'{title}  —  z={z_slice}',
                     color='#e8f5ec', fontsize=13, fontweight='bold', y=1.01)
        save_name= input('Saving Plot| Provide name(e.g: within.png):')
        save_path= os.path.join(self.config.out, save_name)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info(f'Saved → {save_path}')

        plt.show()
        return fig
