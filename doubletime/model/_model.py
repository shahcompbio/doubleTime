import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
import torch
import pickle
import pandas as pd
import anndata as ad
import numpy as np
import random
import wgs_analysis.snvs.mutsig
import logging
import sys


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pyro.set_rng_seed(seed)


class doubleTimeModel(object):

    def __init__(self, adata, tree, snv_types=['1:0', '1:1', '2:0', '2:1', '2:2'], seed=0, lr=0.1, betas=[0.8, 0.99]):
        seed_everything(seed)
        self.adata = adata
        self.tree = tree
        self.snv_types = snv_types

        # TODO: add preprocessing functions that convert adata and tree to the inputs for self.model()
        
        pyro.clear_param_store()
        self.optim = pyro.optim.Adam({'lr': lr, 'betas': betas})
        self.elbo = TraceEnum_ELBO(max_plate_nesting=3)
        self.guide = AutoNormal(
            pyro.poutine.block(self.model, expose=['state_probs']))
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)
        self.losses = []
        self.trace = None

    @config_enumerate
    def model(self, total_cns=None, total_counts=None, alt_counts=None, cn_states=None):

        # Extract the dimensions of the model states
        n_snv_types = cn_states.shape[0]
        n_states = cn_states.shape[1]
        n_clones = cn_states.shape[2]
        n_alleles = cn_states.shape[3]

        # Confirm shapes of the data relative to model states
        if total_cns is not None:
            assert len(total_cns) == n_snv_types
            for a in total_cns:
                assert a.shape[1] == n_clones
        if total_counts is not None:
            assert len(total_counts) == n_snv_types
            for a in total_counts:
                assert a.shape[1] == n_clones
        if alt_counts is not None:
            assert len(alt_counts) == n_snv_types
            for a in alt_counts:
                assert a.shape[1] == n_clones

        # Hyperparameters for prior
        alpha = torch.ones([n_states])
        
        # Sample first categorical variable with a common Dirichlet prior
        state_probs = pyro.sample('state_probs', dist.Dirichlet(alpha))
        
        # loop over all the possible SNV types in a given sample
        # each SNV type is defined by the major and minor copy numbers
        for i, ascn in enumerate(self.snv_types):
            # extract major and minor copy numbers for this SNV type
            major_cn, minor_cn = ascn.split(':')
            major_cn = int(major_cn)
            minor_cn = int(minor_cn)
        
            # extract cn_states and n_snvs for this SNV type
            # the input cn_states and n_snvs are lists of tensors, one for each SNV type
            temp_cn_states = cn_states[i]
            temp_total_counts = total_counts[i]
            temp_total_cn = total_cns[i]
            temp_n_snvs = total_counts[i].shape[0]

            if temp_n_snvs == 0:
                continue

            with pyro.plate(f'data_{major_cn}{minor_cn}', temp_n_snvs, dim=-1):

                # Sample node index
                state_idx = pyro.sample(f'state_idx_{major_cn}{minor_cn}', dist.Categorical(state_probs))

                # sample allele index when considering SNVs that are not LOH
                if minor_cn > 0:
                    allele_idx = pyro.sample(f'allele_idx_{major_cn}{minor_cn}', dist.Categorical(torch.ones([n_alleles]) / n_alleles))
                else:
                    allele_idx = 0  # assume the SNV is coming from the major allele (index 0) when minor_cn = 0

                # loop over all the clones in the tree
                for clone in pyro.plate(f'node_{major_cn}{minor_cn}', n_clones):
                    
                    # True VAF
                    variant_frequency = (temp_cn_states[state_idx, clone, allele_idx] / temp_total_cn[:, clone]) * 0.99 + 0.005

                    # Model for SNV counts
                    p_1 = variant_frequency
                    phi_1 = 100.
                    alpha_1 = p_1 * phi_1
                    beta_1 = (1 - p_1) * phi_1

                    if alt_counts is not None:
                        temp_alt_counts = alt_counts[i]
                        pyro.sample(f'alt_counts_{major_cn}{minor_cn}_{clone}', dist.BetaBinomial(
                            concentration1=alpha_1,
                            concentration0=beta_1,
                            total_count=temp_total_counts[:, clone]), obs=temp_alt_counts[:, clone])

                    else:
                        pyro.sample(f'alt_counts_{major_cn}{minor_cn}_{clone}', dist.BetaBinomial(
                            concentration1=alpha_1,
                            concentration0=beta_1,
                            total_count=temp_total_counts[:, clone]))

    def train(self, n_iter=200):
        for _ in range(n_iter):
            loss = self.svi.step(
                total_cns=self.total_cns, total_counts=self.total_counts, alt_counts=self.alt_counts, cn_states=self.cn_states
            )
            self.losses.append(loss)
        print('Initial loss: {}'.format(self.losses[0]))
        print('Final loss: {}'.format(self.losses[-1]))
    

    def get_model_trace(self):
        # extract fitted values
        # map_estimates = self.guide(
        #     total_cns=self.total_cns, total_counts=self.total_counts, alt_counts=self.alt_counts, cn_states=self.cn_states
        # )
        # learned_state_probs = map_estimates['state_probs'].detach().numpy()

        # learned_state_probs_df = pd.DataFrame({
        #     'learned_state_probs': learned_state_probs,
        # }, index=temp_cn_states_df_a.index).reset_index()

        # Get MAP estimate of node assignments
        guide_trace = poutine.trace(self.guide).get_trace(
            total_cns=self.total_cns, total_counts=self.total_counts, alt_counts=self.alt_counts, cn_states=self.cn_states
        )
        trained_model = poutine.replay(self.model, trace=guide_trace)
        inferred_model = infer_discrete(trained_model, temperature=1, first_available_dim=-3)
        self.trace = poutine.trace(inferred_model).get_trace(
            total_cns=self.total_cns, total_counts=self.total_counts, alt_counts=self.alt_counts, cn_states=self.cn_states
        )
    
    def format_model_output(self, ref_genome):
        if self.trace is None:
            print('Model trace is empty. Running get_model_trace() first.')
            self.get_model_trace()
        
        # Format the learned state assignments to an output dataframe
        data = []
        for i, ascn in enumerate(self.snv_types):
            if len(self.total_counts[i]) == 0:
                continue

            cnA = int(ascn.split(':')[0])
            cnB = int(ascn.split(':')[1])
            temp_learned_state_idx = self.trace.nodes[f'state_idx_{cnA}{cnB}']['value'].detach().numpy()

            temp_data = pd.Series(snv_ids_list[i], name='snv_id').rename_axis('snv').reset_index()
            temp_data = temp_data.merge(pd.DataFrame(alt_counts[i].detach().numpy(), columns=clone_names).melt(ignore_index=False, var_name='leaf', value_name='alt_counts').rename_axis('snv').reset_index())
            temp_data = temp_data.merge(pd.DataFrame(total_counts[i].detach().numpy(), columns=clone_names).melt(ignore_index=False, var_name='leaf', value_name='total_counts').rename_axis('snv').reset_index())
            temp_data = temp_data.merge(pd.DataFrame(cn_states[i, temp_learned_state_idx, :, 0].detach().numpy(), columns=clone_names).melt(ignore_index=False, var_name='leaf', value_name='cn_state_a').rename_axis('snv').reset_index())
            temp_data = temp_data.merge(pd.DataFrame(cn_states[i, temp_learned_state_idx, :, 1].detach().numpy(), columns=clone_names).melt(ignore_index=False, var_name='leaf', value_name='cn_state_b').rename_axis('snv').reset_index())
            # temp_data = temp_data.merge(pd.DataFrame({'cn_state_idx': temp_cn_states_df_a.index.get_level_values('cn_state_idx')[temp_learned_state_idx]}).rename_axis('snv').reset_index())
            temp_data = temp_data.merge(pd.DataFrame({'clade': temp_cn_states_df_a.index.get_level_values('clade')[temp_learned_state_idx]}).rename_axis('snv').reset_index())
            # temp_data = temp_data.merge(pd.DataFrame({'wgd_timing': temp_cn_states_df_a.index.get_level_values('wgd_timing')[temp_learned_state_idx]}).rename_axis('snv').reset_index())
            temp_data['cn_state_a'] = temp_data['cn_state_a'].astype(int).astype('category')
            temp_data['cn_state_b'] = temp_data['cn_state_b'].astype(int).astype('category')
            temp_data['vaf'] = temp_data['alt_counts'] / temp_data['total_counts']

            if data != []:
                # reindex the snv count based on the highest snv count in the previous SNV type
                temp_data['snv'] = temp_data['snv'] + data[-1]['snv'].max() + 1
            
            # add a column for the SNV type
            temp_data['ascn'] = ascn
            
            if len(temp_data) > 0:
                data.append(temp_data)

        data = pd.concat(data, ignore_index=True)
        parts = data.snv_id.str.split(':', expand = True)
        data['chromosome'] = parts.iloc[:, 0]
        data['position'] = parts.iloc[:, 1].astype(int)

        assert len(data) == len(parts)

        snv_counts = data[['snv', 'clade']].drop_duplicates().groupby(['clade']).size()
        snv_counts = snv_counts.reindex(index=learned_state_probs_df.set_index(['clade']).index, fill_value=0)

        # look at APOBEC on the tree
        data['ref'] = data.snv_id.str.split(':', expand=True).iloc[:, 2].str.split('>', expand=True).iloc[:, 0]
        data['alt'] = data.snv_id.str.split(':', expand=True).iloc[:, 2].str.split('>', expand=True).iloc[:, 1]
        data['chrom'] = data['chromosome']
        data['coord'] = data['position']
        wgs_analysis.snvs.mutsig.calculate_tri_nucleotide_context(data, ref_genome)
        data['is_apobec'] = [dt.tl.is_apobec_snv(r.ref, r.alt, r.tri_nucleotide_context) for _, r in data.iterrows()]

        data['is_cpg'] = dt.tl.is_cpg_snv(data)

        data['is_cpg2'] = [dt.tl.is_c_to_t_in_cpg_context(r.ref, r.alt, r.tri_nucleotide_context) for _, r in data.iterrows()]
        assert data.is_cpg2.equals(data.is_cpg)
        data = data.drop(columns=['is_cpg2'])
        
        multi_assigned = [sdf for x, sdf in data.groupby('snv_id') if len(sdf.clade.unique()) > 1]
        assert len(multi_assigned) == 0, (f'{len(multi_assigned)}/{len(data.snv_id.unique())} SNVs are assigned to multiple clades')

        data.to_csv(output, index = False)
        