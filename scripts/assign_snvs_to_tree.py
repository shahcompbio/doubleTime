import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
import torch
import pickle
import click
import pandas as pd
import anndata as ad
import numpy as np
import wgs_analysis.snvs.mutsig
import logging
import sys


@config_enumerate
def model(total_cns=None, total_counts=None, alt_counts=None, cn_states=None, snv_types=['2:0', '2:1']):

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
    for i, ascn in enumerate(snv_types):
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


def build_cn_states_df(tree, cnA, cnB):
    '''
    Build a dataframe of all the possible CN states for each clone in the tree.
    This function currently assumes that there is at most one WGD event in the tree.
    Thus it only accounts for the following allele copy number states: 2|0, 2|1, 2|2, 1|1, 1|0.
    ----------
    tree: Bio.Phylo.BaseTree.Tree
        The tree of clades. Each clade should have a boolean attribute 'is_wgd' indicating whether there is a WGD event at this branch.
        There should be leafs attached to each clade. Each leaf should have an integer attribute 'n_wgd' indicating the number of WGD events and the name of the clone.
    cnA: int
        The copy number of the A allele.
    cnB: int
        The copy number of the B allele.
    '''

    def calculate_snv_multiplicity(cn, leaf_n_wgd, cladename):
        wgd_timing = 'post' if cladename.startswith('post') else 'pre'
        if cn == 0:
            return 0
        elif wgd_timing == 'pre' and leaf_n_wgd == 1:
            return cn
        else:
            return 1

    cn_states_df = []
    for clade in tree.find_clades():
        assert clade.n_wgd <= 1
        for leaf in clade.get_terminals():
            cn_states_df.append({
                'clade': clade.name,
                'leaf': leaf.name,
                'cn_a': calculate_snv_multiplicity(cnA, leaf.n_wgd, clade.name),
                'cn_b': calculate_snv_multiplicity(cnB, leaf.n_wgd, clade.name)})

    # Add zero state to account for the case where the SNV is assigned outside the tree
    for leaf in tree.get_terminals():
        cn_states_df.append({
            'clade': 'none',
            'leaf': leaf.name,
            'cn_a': 0,
            'cn_b': 0})
        
    cn_states_dfa = pd.DataFrame(cn_states_df).set_index(['clade', 'leaf'])['cn_a'].unstack(fill_value=0)
    cn_states_dfb = pd.DataFrame(cn_states_df).set_index(['clade', 'leaf'])['cn_b'].unstack(fill_value=0)

    return cn_states_dfa, cn_states_dfb


def is_apobec_snv(ref_base, alt_base, trinucleotide_context):
    """
    Classify a SNV as APOBEC-induced based on its substitution type and trinucleotide context.
    This function also accounts for the reverse complement context.

    Parameters:
    - ref_base: The reference base (e.g., 'C').
    - alt_base: The alternate base (e.g., 'T').
    - trinucleotide_context: The trinucleotide context (e.g., 'TCA').

    Returns:
    - True if the SNV is APOBEC-induced, False otherwise.
    """

    # Check if the substitution is a C-to-T or G-to-A transition
    is_c_to_t = ref_base.upper() == 'C' and alt_base.upper() == 'T'
    is_g_to_a = ref_base.upper() == 'G' and alt_base.upper() == 'A'

    # Check if the substitution is a C-to-G or G-to-C transition
    is_c_to_g = ref_base.upper() == 'C' and alt_base.upper() == 'G'
    is_g_to_c = ref_base.upper() == 'G' and alt_base.upper() == 'C'

    # Check if the trinucleotide context fits the TpCpX pattern on the forward strand
    is_tpctx_forward = trinucleotide_context[1].upper() == 'C' and trinucleotide_context[0].upper() == 'T'

    # Check if the trinucleotide context fits the RpGpX pattern on the reverse strand (where R is A or G)
    is_tpctx_reverse = trinucleotide_context[1].upper() == 'G' and trinucleotide_context[2].upper() == 'A'

    # APOBEC-induced mutations are C-to-T or C-to-G in TpCpX context or reverse complement
    return ((is_c_to_t or is_c_to_g) and is_tpctx_forward) or ((is_g_to_a or is_g_to_c) and is_tpctx_reverse)


def is_c_to_t_in_cpg_context(ref_base, alt_base, trinucleotide_context):
    """
    This function checks if a single nucleotide variant (SNV) is a C to T mutation
    in a CpG context or its reverse complement G to A in a CpG context.
    
    Parameters:
    ref_base (str): The reference nucleotide
    alt_base (str): The alternate nucleotide
    trinucleotide_context (str): The trinucleotide context of the SNV (string of 3 nucleotides)
    
    Returns:
    bool: True if the mutation is a C to T mutation in a CpG context or a G to A mutation
          in a CpG context on the reverse strand, False otherwise.
    """
    
    # Check if the mutation is C to T in a CpG context on the forward strand
    if ref_base == 'C' and alt_base == 'T':
        if len(trinucleotide_context) == 3 and trinucleotide_context[1] == 'C' and trinucleotide_context[2] == 'G':
            return True

    # Check if the mutation is G to A in a CpG context on the reverse strand
    if ref_base == 'G' and alt_base == 'A':
        if len(trinucleotide_context) == 3 and trinucleotide_context[0] == 'C' and trinucleotide_context[1] == 'G':
            return True
    
    return False

def write_empty(output):
    df = pd.DataFrame(columns=[
        'snv', 'leaf', 'alt_counts', 'total_counts', 'cn_state_a', 'cn_state_b', 'clade',
        'vaf', 'snv_id', 'ascn', 'chromosome', 'position', 'ref', 'alt', 'chrom', 'coord', 'tri_nucleotide_context', 
        'is_apobec', 'is_cpg'])
    df.to_csv(output, index=False)


@click.command()
@click.option('--adata')
@click.option('--tree')
@click.option('--output')
@click.option('--ref_genome')
@click.option('--min_total_counts_perblock', type=int, default=2, required=False)
@click.option('--cnloh_only', is_flag=True, default=False, help="Include only SNVs in cnLOH regions")
def main(adata, tree, output, ref_genome, min_total_counts_perblock, cnloh_only):
    if cnloh_only:
        snv_types = ['2:0']
    else:
        snv_types = ['1:0', '1:1', '2:0', '2:1', '2:2']

    adata = ad.read_h5ad(adata)

    # Remove non-homogenous snvs
    adata = adata[:, adata.var['is_homogenous_cn']].copy()

    if np.min(adata.shape) == 0:
        print("Anndata has 0 clones or SNVs, shape=", adata.shape)
        write_empty(output)
        return

    tree = pickle.load(open(tree, 'rb'))

    # restrict to those SNVs with sufficient covering reads in all clones
    adata.var['min_total_count'] = np.array(adata.layers['total_count'].min(axis=0))    
    print(f'Number of SNVs with at least {min_total_counts_perblock} reads in all clones:', 
        (adata.var['min_total_count'] >= min_total_counts_perblock).sum())
    adata = adata[:, adata.var['min_total_count'] >= min_total_counts_perblock]

    # restrict anndata to clones represented in the tree
    clones = []
    for leaf in tree.get_terminals():
        clones.append(leaf.name.replace('clone_', '').replace('postwgd_', ''))
    adata = adata[clones].copy()

    # Fixed order for clone names
    clade_names = [clade.name for clade in tree.find_clades()]
    # set clone name to postwgd_clone_{a} if we had to split a terminal branch into two due to a WGD event, otherwise set it to clone_{a}
    clone_names = [f'postwgd_clone_{a}' if f'postwgd_clone_{a}' in clade_names else f'clone_{a}' for a in adata.obs.index]

    # Fixed order for Leaf ids
    leaf_ids = adata.obs.index

    # prepare input to model
    cn_states_list = []
    total_counts_list = []
    alt_counts_list = []
    total_cn_list = []
    snv_ids_list = []
    for ascn in snv_types:
        cnA = int(ascn.split(':')[0])
        cnB = int(ascn.split(':')[1])
        temp_cn_states_df_a, temp_cn_states_df_b = build_cn_states_df(tree, cnA, cnB)

        # Ensure correct ordering
        temp_cn_states_df_a = temp_cn_states_df_a[clone_names]
        temp_cn_states_df_b = temp_cn_states_df_b[clone_names]
        temp_cn_states = torch.stack([torch.tensor(temp_cn_states_df_a.values), torch.tensor(temp_cn_states_df_b.values)], dim=2)
        cn_states_list.append(temp_cn_states)

        temp_adata = adata[leaf_ids, (adata.var['snv_type'] == ascn)].copy()
        temp_total_counts = torch.tensor(np.array(temp_adata.layers['total_count']).T, dtype=torch.float32)
        temp_alt_counts = torch.tensor(np.array(temp_adata.layers['alt_count']).T, dtype=torch.float32)
        temp_total_cn = torch.tensor(np.array(temp_adata.layers['Maj']).T + np.array(temp_adata.layers['Min']).T, dtype=torch.float32)
        total_counts_list.append(temp_total_counts)
        alt_counts_list.append(temp_alt_counts)
        total_cn_list.append(temp_total_cn)
        snv_ids_list.append(temp_adata.var.index)

    # stack the list of tensors (one for each SNV type) into a single tensor
    cn_states = torch.stack(cn_states_list, dim=0)


    # Ensure there is at least one snv
    n_snvs = sum(a.shape[0] for a in total_counts_list)
    if n_snvs == 0:
        print(f"No clonal SNVs found in allowed CN states ({snv_types}), writing empty table.")
        write_empty(output)
        return

    total_counts = total_counts_list
    alt_counts = alt_counts_list
    total_cns = total_cn_list

    # fit model
    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=3)

    pyro.set_rng_seed(1)
    pyro.clear_param_store()

    guide = AutoNormal(
        pyro.poutine.block(model, expose=['state_probs']))

    svi = SVI(model, guide, optim, loss=elbo)

    losses = []
    for i in range(200):
        loss = svi.step(
            total_cns=total_cns, total_counts=total_counts, alt_counts=alt_counts, cn_states=cn_states, snv_types=snv_types
        )
        losses.append(loss)
    
    final_loss = svi.evaluate_loss(
        total_cns=total_cns, total_counts=total_counts, alt_counts=alt_counts, cn_states=cn_states, snv_types=snv_types
    )
    print("initial loss:", losses[0])
    print('final loss:', final_loss)


    # extract fitted values
    map_estimates = guide(
        total_cns=total_cns, total_counts=total_counts, alt_counts=alt_counts, cn_states=cn_states, snv_types=snv_types
    )
    learned_state_probs = map_estimates['state_probs'].detach().numpy()

    learned_state_probs_df = pd.DataFrame({
        'learned_state_probs': learned_state_probs,
    }, index=temp_cn_states_df_a.index).reset_index()

    # Get MAP estimate of node assignments
    guide_trace = poutine.trace(guide).get_trace(
        total_cns=total_cns, total_counts=total_counts, alt_counts=alt_counts, cn_states=cn_states, snv_types=snv_types
    )
    trained_model = poutine.replay(model, trace=guide_trace)
    inferred_model = infer_discrete(trained_model, temperature=1, first_available_dim=-3)
    trace = poutine.trace(inferred_model).get_trace(
        total_cns=total_cns, total_counts=total_counts, alt_counts=alt_counts, cn_states=cn_states, snv_types=snv_types
    )

    # Format the learned state assignments to an output dataframe
    data = []
    for i, ascn in enumerate(snv_types):
        if len(total_counts[i]) == 0:
            continue

        cnA = int(ascn.split(':')[0])
        cnB = int(ascn.split(':')[1])
        temp_learned_state_idx = trace.nodes[f'state_idx_{cnA}{cnB}']['value'].detach().numpy()

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
    data['is_apobec'] = [is_apobec_snv(r.ref, r.alt, r.tri_nucleotide_context) for _, r in data.iterrows()]

    data['is_cpg'] = np.logical_or(np.logical_and(data.ref == 'C', np.logical_and(data.alt == 'T', data.tri_nucleotide_context.str.slice(2) == 'G')),
                                np.logical_and(data.ref == 'G', np.logical_and(data.alt == 'A', data.tri_nucleotide_context.str.slice(0,1) == 'C')))
    data['is_cpg2'] = [is_c_to_t_in_cpg_context(r.ref, r.alt, r.tri_nucleotide_context) for _, r in data.iterrows()]
    assert data.is_cpg2.equals(data.is_cpg)
    data = data.drop(columns=['is_cpg2'])
    
    multi_assigned = [sdf for x, sdf in data.groupby('snv_id') if len(sdf.clade.unique()) > 1]
    assert len(multi_assigned) == 0, (f'{len(multi_assigned)}/{len(data.snv_id.unique())} SNVs are assigned to multiple clades')

    data.to_csv(output, index = False)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
