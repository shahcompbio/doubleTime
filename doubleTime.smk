import os

configfile: "config.yaml"

# input files
snv_adata = config['snv_adata']
cna_adata = config['cna_adata']
cell_info = config['cell_info']

# output directory
outdir = config["outdir"]
patient_id = config["patient_id"]

# parameters
tree_snv_min_clone_size = config['tree_snv_min_clone_size']
tree_snv_min_num_snvs = config['tree_snv_min_num_snvs']
tree_snv_min_prop_clonal_wgd = config['tree_snv_min_prop_clonal_wgd']
genome_fasta_filename = config['genome_fasta_filename']

if not os.path.exists(outdir):
    os.makedirs(outdir)

rule all:
    input: 
        table=os.path.join(outdir, f"{patient_id}_tree_snv_assignment.csv")

rule infer_sbmclone_tree:
    input:
        snv_adata=snv_adata,
    params: 
        patient_id=patient_id,
    output:
        os.path.join(outdir, f"{patient_id}_tree.pickle"),
    shell:
        """
        python scripts/infer_sbmclone_tree.py --snv_adata {input.snv_adata} --patient_id {params.patient_id} --output {output}
        """

# TODO: apply test for independent WGD events and integrate into next step
#rule test_indep_wgd

rule construct_clustered_snv_adata:
    input:
        cna_adata=cna_adata,
        snv_adata=snv_adata,
        cell_info_filename=cell_info,
        tree=os.path.join(outdir, f"{patient_id}_tree.pickle"),
    params: 
        patient_id=patient_id,
    resources:
        mem_mb=64000
    output:
        clustered_snv_adata=os.path.join(outdir, f"{patient_id}_snv_clustered.h5"),
        clustered_cna_adata=os.path.join(outdir, f"{patient_id}_cna_clustered.h5"),
        pruned_tree=os.path.join(outdir, f"{patient_id}_annotated_tree.pickle"),
    shell:
        """
        python scripts/construct_clustered_snv_adata.py \
            --adata_cna {input.cna_adata}  --adata_snv {input.snv_adata} --tree_filename {input.tree} --cell_info_filename {input.cell_info_filename} --patient_id {params.patient_id} \
            --min_clone_size {tree_snv_min_clone_size} --min_num_snvs {tree_snv_min_num_snvs} --min_prop_clonal_wgd {tree_snv_min_prop_clonal_wgd} \
            --output_cn {output.clustered_cna_adata} --output_snv {output.clustered_snv_adata} --output_pruned_tree {output.pruned_tree}
        """

rule assign_snvs_to_tree:
    input:
        tree=os.path.join(outdir, f"{patient_id}_annotated_tree.pickle"),
        adata=os.path.join(outdir, f"{patient_id}_snv_clustered.h5"),
    params: 
        patient_id=patient_id,
    output:
        table=os.path.join(outdir, f"{patient_id}_tree_snv_assignment.csv")
    shell:
        """
        python scripts/assign_snvs_to_tree.py --adata {input.adata} --tree {input.tree} --ref_genome {genome_fasta_filename} --output {output.table}
        """

