import pickle as pkl
import anndata as ad
import torch.nn as nn


class GeneExpModel(nn.Module):
    def __init__(self, slices, label="subclass"):
        super(GeneExpModel, self).__init__()
        self.slices = slices
        self.label = label
        self.num_tokens = 0

    def fit(self):
        concatenated_slices = ad.concat(self.slices)
        unique_subclasses = concatenated_slices.obs[self.label].unique()
        self.num_tokens = len(unique_subclasses)
        try:
            self.id_to_subclass = pkl.load(
                open(
                    "id_to_subclass.pkl",
                    "rb",
                )
            )
            subclass_to_id = {v: k for k, v in self.id_to_subclass.items()}
        except Exception as e:
            print(f"Error loading token map: {e}")
            subclass_to_id = {
                subclass: i for i, subclass in enumerate(unique_subclasses)
            }
            self.id_to_subclass = {
                i: subclass for i, subclass in enumerate(unique_subclasses)
            }
            print("Recreating token map")
        self.subclass_to_id = subclass_to_id
        # Map subclass labels to token IDs
        concatenated_slices.obs["token"] = concatenated_slices.obs[self.label].map(
            subclass_to_id
        )

        # Find NaNs
        nan_mask = concatenated_slices.obs["token"].isna()
        num_nans = nan_mask.sum()

        if num_nans > 0:
            print(
                f"⚠️ Found {num_nans} unmapped tokens after mapping '{self.label}' → 'token'"
            )

            # Count clusters only where token is NaN
            cluster_counts = concatenated_slices.obs.loc[
                nan_mask, "cluster"
            ].value_counts()
            cluster_counts = cluster_counts[cluster_counts > 0]
            print("Clusters with unmapped tokens:")
            print(cluster_counts)

            # Assign the majority (most common) token to these cells
            majority_token = concatenated_slices.obs["token"].mode()[0]
            concatenated_slices.obs.loc[nan_mask, "token"] = majority_token
            print(f"Reassigned missing tokens to majority class: {majority_token}")

        # Ensure integer dtype
        concatenated_slices.obs["token"] = concatenated_slices.obs["token"].astype(int)

        self.concatenated_slices = concatenated_slices

    def get_gene_exp_from_token(self, tokens):
        return [self.token_to_gene_exp[token] for token in tokens]

    def get_label_from_token(self, tokens):
        return [self.id_to_subclass[token] for token in tokens]

    def get_token_from_gene_exp(self, gene_exps):
        return [self.gene_exp_to_token[gene_exp] for gene_exp in gene_exps]

    def get_tokenized_slices(self):
        slices_new = []
        c = 0
        for slice in self.slices:
            slices_new.append(self.concatenated_slices[c : c + len(slice)])
            c += len(slice)
        return slices_new
