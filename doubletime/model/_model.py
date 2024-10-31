import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
import torch
import pandas as pd
import numpy as np
import random
import wgs_analysis.snvs.mutsig
import doubletime as dt
import xarray as xr


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

    def __init__(self, adata, tree, snv_types=['1:0', '1:1', '2:0', '2:1', '2:2'], min_total_counts_perblock=2, seed=0, lr=0.1, betas=[0.8, 0.99]):
        '''
        Initialize the doubleTime model object given the input data.

        Parameters
        ----------
        adata : anndata.AnnData
            Anndata object containing the SNV data. Observations (.obs, matrix rows) in this object should correspond to clones (leafs in the tree) 
            and variables (.var, matrix columns) should correspond to unique SNVs.
            This object should contain the following layers:
                total_count : total read counts for each SNV
                alt_count : alternate read counts for each SNV
                Maj: major copy number for each SNV
                Min: minor copy number for each SNV
            This object should contain the following variables:
                snv_type : SNV type for each SNV
        tree : Bio.Phylo.Tree
            Phylogenetic tree object containing the clone relationships.
        snv_types : list of str
            List of SNV types to consider in the model. Each SNV type should be formatted as 'major_cn:minor_cn'.
        min_total_counts_perblock : int
            Minimum total read counts required for an SNV to be considered in the model.
        seed : int
            Random seed for reproducibility.
        lr : float
            Learning rate for the model.
        betas : list of float
            Betas for the Adam optimizer.

        '''
        seed_everything(seed)
        self.adata = adata
        self.tree = tree
        self.snv_types = snv_types
        self.min_total_counts_perblock = min_total_counts_perblock

        # preprocessing function that convert adata and tree to the inputs for self.model()
        self.total_cns, self.total_counts, self.alt_counts, \
            self.cn_states, self.snv_ids, self.clone_names, self.clade_index = self.preprocess_data()
        
        # Initialize the model
        pyro.clear_param_store()
        self.optim = pyro.optim.Adam({'lr': lr, 'betas': betas})
        self.elbo = TraceEnum_ELBO(max_plate_nesting=3)
        self.guide = AutoNormal(
            pyro.poutine.block(self.model, expose=['state_probs']))
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.elbo)

        # Initialize variables that will be used after training the model
        self.losses = []
        self.trace = None

    @config_enumerate
    def model(self, total_cns=None, total_counts=None, alt_counts=None, cn_states=None):
        '''
        Pyro model function for the doubleTime model. This function defines the generative process for the model.

        Parameters
        ----------
        total_cns : list of tensors
            List of tensors containing the total copy number of each SNV. One tensor per SNV type.
        total_counts : list of tensors
            List of tensors containing the total read counts of each SNV. One tensor per SNV type.
        alt_counts : list of tensors
            List of tensors containing the alternate read counts of each SNV. One tensor per SNV type.
        cn_states : tensor
            Tensor containing the possible copy number states for each SNV in each clone and each SNV type 
        '''

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
        '''
        Train the model using a specified number of iterations. Losses are stored in self.losses. Posterior estimates
        can be obtained using self.get_model_trace() after training.

        Parameters
        ----------
        n_iter : int
            Number of iterations to train the model
        '''
        for _ in range(n_iter):
            loss = self.svi.step(
                total_cns=self.total_cns, total_counts=self.total_counts, alt_counts=self.alt_counts, cn_states=self.cn_states
            )
            self.losses.append(loss)
        print('Initial loss: {}'.format(self.losses[0]))
        print('Final loss: {}'.format(self.losses[-1]))

    def get_model_trace(self):
        '''
        Get the model trace after training the model. These are the posterior estimates of the model parameters.
        Outputs are stored in self.trace rather than returned.
        '''
        # Get MAP estimate of node assignments
        guide_trace = poutine.trace(self.guide).get_trace(
            total_cns=self.total_cns, total_counts=self.total_counts, alt_counts=self.alt_counts, cn_states=self.cn_states
        )
        trained_model = poutine.replay(self.model, trace=guide_trace)
        inferred_model = infer_discrete(trained_model, temperature=1, first_available_dim=-3)
        self.trace = poutine.trace(inferred_model).get_trace(
            total_cns=self.total_cns, total_counts=self.total_counts, alt_counts=self.alt_counts, cn_states=self.cn_states
        )

    def preprocess_data(self):
        '''
        Preprocess the adata and tree data to create input parameters for the pyro model function (self.model)

        Returns
        -------
        total_cns : list of tensors
            List of tensors containing the total copy number of each SNV. One tensor per SNV type.
        total_counts : list of tensors
            List of tensors containing the total read counts of each SNV. One tensor per SNV type.
        alt_counts : list of tensors
            List of tensors containing the alternate read counts of each SNV. One tensor per SNV type.
        cn_states : tensor
            Tensor containing the possible copy number states for each SNV in each clone and each SNV type
        snv_ids : list of lists
            List of lists containing the SNV ids for each SNV type.
        clone_names : list
            List of clone names in the tree
        '''
        # restrict to those SNVs with sufficient covering reads in all clones
        self.adata.var['min_total_count'] = np.array(self.adata.layers['total_count'].min(axis=0))    
        print(f'Number of SNVs with at least {self.min_total_counts_perblock} reads in all clones:', 
            (self.adata.var['min_total_count'] >= self.min_total_counts_perblock).sum())
        self.adata = self.adata[:, self.adata.var['min_total_count'] >= self.min_total_counts_perblock]

        # restrict anndata to clones represented in the tree
        clones = []
        for leaf in self.tree.get_terminals():
            clones.append(leaf.name.replace('clone_', '').replace('postwgd_', ''))
        self.adata = self.adata[clones].copy()

        # Fixed order for clone names
        clade_names = [clade.name for clade in self.tree.find_clades()]
        # set clone name to postwgd_clone_{a} if we had to split a terminal branch into two due to a WGD event, otherwise set it to clone_{a}
        clone_names = [f'postwgd_clone_{a}' if f'postwgd_clone_{a}' in clade_names else f'clone_{a}' for a in self.adata.obs.index]

        # Fixed order for Leaf ids
        leaf_ids = self.adata.obs.index

        # prepare input to model
        cn_states_list = []
        total_counts_list = []
        alt_counts_list = []
        total_cn_list = []
        snv_ids_list = []
        # loop over all the possible SNV types in a given sample
        for ascn in self.snv_types:
            cnA = int(ascn.split(':')[0])
            cnB = int(ascn.split(':')[1])
            # build the copy number states DataFrame for the current SNV type
            temp_cn_states_df_a, temp_cn_states_df_b = dt.tl.build_cn_states_df(self.tree, cnA, cnB)

            # Ensure correct ordering of clones
            temp_cn_states_df_a = temp_cn_states_df_a[clone_names]
            temp_cn_states_df_b = temp_cn_states_df_b[clone_names]

            # convert the DataFrames to DataArrays
            # the DataArray will have dimensions of tree position (state), clone, and allele
            temp_cn_states = xr.DataArray(
                np.stack([temp_cn_states_df_a.values, temp_cn_states_df_b.values], axis=2),
                dims=["state", "clone", "allele"],
                coords={"state": temp_cn_states_df_a.index.values, "clone": leaf_ids.values, "allele": ["a", "b"]}
            )
            # append the DataArray to the list
            # no need to convert to a tensor yet since the list of DataArrays will be converted to a tensor later
            cn_states_list.append(temp_cn_states)

            # subset adata to just the SNVs of the current SNV type
            temp_adata = self.adata[leaf_ids, (self.adata.var['snv_type'] == ascn)].copy()
            temp_snv_ids = temp_adata.var.index.values

            # convert the total counts, alt counts, and total CN to DataArrays with the dimensions of snv_id and clone
            # importantly, the length of the snv_id dimension will vary as we loop over SNV types
            temp_total_counts = xr.DataArray(
                np.array(temp_adata.layers['total_count']).T,
                dims=["snv_id", "clone"],
                coords={"snv_id": temp_snv_ids, "clone": leaf_ids.values}
            )
            temp_alt_counts = xr.DataArray(
                np.array(temp_adata.layers['alt_count']).T,
                dims=["snv_id", "clone"],
                coords={"snv_id": temp_snv_ids, "clone": leaf_ids.values}
            )
            temp_total_cn = xr.DataArray(
                np.array(temp_adata.layers['Maj']).T + np.array(temp_adata.layers['Min']).T,
                dims=["snv_id", "clone"],
                coords={"snv_id": temp_snv_ids, "clone": leaf_ids.values}
            )

            # we convert these DataArrays to tensors before appending to the list since they cannot be stacked along the snv_id dimension,
            # unlike the cn_states DataArrays, which means we pass these objects to the model as a list of tensors
            total_counts_list.append(torch.tensor(temp_total_counts.values, dtype=torch.float32))
            alt_counts_list.append(torch.tensor(temp_alt_counts.values, dtype=torch.float32))
            total_cn_list.append(torch.tensor(temp_total_cn.values, dtype=torch.float32))
            snv_ids_list.append(temp_snv_ids)

        # stack the list of cn states DataArrays (one for each SNV type) into a single DataArray and convert to a tensor
        cn_states = torch.tensor(xr.concat(cn_states_list, dim="snv_type").values, dtype=torch.float32)

        # Ensure there is at least one snv
        n_snvs = sum(a.shape[0] for a in total_counts_list)
        if n_snvs == 0:
            raise ValueError(f"No clonal SNVs found in allowed CN states ({self.snv_types}).")
        
        clade_index = temp_cn_states_df_a.index.get_level_values('clade')

        return total_cn_list, total_counts_list, alt_counts_list, cn_states, snv_ids_list, clone_names, clade_index

    def format_model_output(self, ref_genome=None):
        '''
        Format the model output into a pandas dataframe. This dataframe contains the SNV ids, the SNV type, the clone
        name, the total read counts, the alternate read counts, the copy number states, the clade assignment, and the VAF.

        Parameters
        ----------
        ref_genome : str
            Path to the reference genome. If provided, the tri_nucleotide_context will be calculated for each SNV.

        Returns
        -------
        data : pandas.DataFrame
            Formatted output dataframe containing the SNV ids, the SNV type, the clone name, the total read counts, the
            alternate read counts, the copy number states, the clade assignment determined by the pyro model, and the VAF.
        '''
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

            # find out which branch each SNV is assigned to for all SNVs belonging to this SNV type
            temp_learned_state_idx = self.trace.nodes[f'state_idx_{cnA}{cnB}']['value'].detach().numpy()

            # create a DataArray with the SNV ids for this SNV type
            temp_data = xr.DataArray(self.snv_ids[i], dims=["snv"], name='snv_id')

            # create DataArrays for the alt counts, total counts, copy number states, and clade index for this SNV type
            snv_ids = range(self.alt_counts[i].shape[0])
            alt_counts_da = xr.DataArray(self.alt_counts[i].detach().numpy(), dims=["snv", "leaf"], coords={"snv": snv_ids, "leaf": self.clone_names})
            total_counts_da = xr.DataArray(self.total_counts[i].detach().numpy(), dims=["snv", "leaf"], coords={"snv": snv_ids, "leaf": self.clone_names})
            cn_state_a_da = xr.DataArray(self.cn_states[i, temp_learned_state_idx, :, 0].detach().numpy(), dims=["snv", "leaf"], coords={"snv": snv_ids, "leaf": self.clone_names})
            cn_state_b_da = xr.DataArray(self.cn_states[i, temp_learned_state_idx, :, 1].detach().numpy(), dims=["snv", "leaf"], coords={"snv": snv_ids, "leaf": self.clone_names})
            clade_da = xr.DataArray(self.clade_index[temp_learned_state_idx], dims=["snv"], coords={"snv": range(len(self.clade_index[temp_learned_state_idx]))})

            # merge the DataArrays into one DataArray, using their shared 'snv' and 'leaf' coordinates to merge
            temp_data = xr.merge([temp_data, alt_counts_da.rename('alt_counts'), total_counts_da.rename('total_counts'), 
                                  cn_state_a_da.rename('cn_state_a'), cn_state_b_da.rename('cn_state_b'), clade_da.rename('clade')])

            # convert the copy number states to integers
            temp_data['cn_state_a'] = temp_data['cn_state_a'].astype(int)
            temp_data['cn_state_b'] = temp_data['cn_state_b'].astype(int)
            # compute the VAF based on the total and alternate read counts
            temp_data['vaf'] = temp_data['alt_counts'] / temp_data['total_counts']

            if data != []:
                # reindex the snv count based on the highest snv count in the previous SNV type
                temp_data['snv'] = temp_data['snv'] + data[-1]['snv'].max() + 1
            
            # add a variable for the SNV type
            temp_data = temp_data.assign_coords(ascn=ascn)
            
            if len(temp_data['snv']) > 0:
                data.append(temp_data)

        data = xr.concat(data, dim='snv')
        parts = xr.DataArray(
            np.array([snv_id.split(':') for snv_id in data['snv_id'].values]),
            dims=["snv", "part"]
        )
        data = data.assign_coords(chromosome=parts[:, 0], position=parts[:, 1].astype(int))

        assert len(data['snv']) == len(parts)

        # convert the DataArray to a pandas DataFrame
        # this is necessary as wgs_analysis tools for calculating tri_nucleotide_context require a DataFrame
        data = data.to_dataframe().reset_index(drop=False)

        # only look at tri_nucleotide_context if a reference genome is provided
        if ref_genome is not None:
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
        
        # check if any SNVs are assigned to multiple clades
        multi_assigned = [sdf for x, sdf in data.groupby('snv_id') if len(sdf.clade.unique()) > 1]
        assert len(multi_assigned) == 0, (f'{len(multi_assigned)}/{len(data.snv_id.unique())} SNVs are assigned to multiple clades')

        return data
        