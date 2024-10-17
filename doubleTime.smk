import os

configfile: "config.yaml"

# input files
snv_adata = config['snv_adata']
cna_adata = config['cna_adata']

# output directory
outdir = config["outdir"]
patient_id = config["patient_id"]

# parameters
tree_snv_min_clone_size = config['tree_snv_min_clone_size']
tree_snv_min_num_snvs = config['tree_snv_min_num_snvs']
tree_snv_min_prop_clonal_wgd = config['tree_snv_min_prop_clonal_wgd']
genome_fasta_filename = config['genome_fasta_filename']

if 'binarization_threshold' in config:
    binarization_threshold = config['binarization_threshold']
else:
    binarization_threshold = 0.01

repo_dir = '/data1/shahs3/users/weinera2/doubleTime'

scripts_dir = os.path.join(repo_dir, 'scripts')
# tree_snv_qc_template = os.path.join(repo_dir, 'templates', 'qc_clone_tree.md')


if not os.path.exists(outdir):
    os.makedirs(outdir)

rule all:
    input: 
        os.path.join(outdir, f"{patient_id}_tree_snv_assignment.csv"),
        os.path.join(outdir, f"{patient_id}_snv_reads_hist.pdf")

rule infer_sbmclone_tree:
    input:
        snv_adata=snv_adata,
    params: 
        patient_id=patient_id,
        binarization_threshold=config['binarization_threshold'],
    resources:
        mem_mb=64000
    output:
        os.path.join(outdir, f"{patient_id}_tree.pickle"),
    shell:
        """
        python scripts/infer_sbmclone_tree.py --snv_adata {input.snv_adata} --patient_id {params.patient_id} --output {output} \
            --binarization_threshold {binarization_threshold}
        """

# TODO: apply test for independent WGD events and integrate into next step
#rule test_indep_wgd

rule construct_clustered_snv_adata:
    input:
        cna_adata=cna_adata,
        snv_adata=snv_adata,
        tree=os.path.join(outdir, f"{patient_id}_tree.pickle"),
    params: 
        patient_id=patient_id,
    resources:
        mem_mb=128000
    output:
        clustered_snv_adata=os.path.join(outdir, f"{patient_id}_snv_clustered.h5"),
        clustered_cna_adata=os.path.join(outdir, f"{patient_id}_cna_clustered.h5"),
        pruned_tree=os.path.join(outdir, f"{patient_id}_annotated_tree.pickle"),
    shell:
        """
        python {scripts_dir}/construct_clustered_snv_adata.py \
            --adata_cna {input.cna_adata}  --adata_snv {input.snv_adata} --tree_filename {input.tree} \
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
        python {scripts_dir}/assign_snvs_to_tree.py --adata {input.adata} --tree {input.tree} --ref_genome {genome_fasta_filename} --output {output.table}
        """


rule qc_output_plots:
    input:
        tree=os.path.join(outdir, f"{patient_id}_annotated_tree.pickle"),
        adata=os.path.join(outdir, f"{patient_id}_snv_clustered.h5"),
        table=os.path.join(outdir, f"{patient_id}_tree_snv_assignment.csv")
    params:
        patient_id=patient_id
    output:
        snv_reads_hist = os.path.join(outdir, f"{patient_id}_snv_reads_hist.pdf"),
    shell:
        """
        python {scripts_dir}/plot_qc_output.py \
        --adata_filename {input.adata} --tree_filename {input.tree} --table_filename {input.table} --patient_id {params.patient_id} \
        -srh {output.snv_reads_hist}
        """

# rule qc_notebook:
#     input:
#         tree=os.path.join(outdir, f"{patient_id}_annotated_tree.pickle"),
#         adata=os.path.join(outdir, f"{patient_id}_snv_clustered.h5"),
#         table=os.path.join(outdir, f"{patient_id}_tree_snv_assignment.csv"),
#         template=tree_snv_qc_template,
#     params:
#         patient_id="{patient_id}",
#         job_dir=outdir
#     output:
#         notebook=os.path.join(outdir, "{patient_id}_qc_notebook.ipynb"),
#     shell:
#         """
#         mkdir -p {params.job_dir}
#         cd {params.job_dir}
#         python {scripts_dir}/render_myst.py {input.template} temp_{params.patient_id}.md '{{"patient_id": "{params.patient_id}"}}'
#         jupytext temp_{params.patient_id}.md --from myst --to ipynb --output temp_{params.patient_id}_template.ipynb
#         python -m papermill -p patient_id {params.patient_id} -p adata_filename {input.adata} -p tree_filename {input.tree} -p table_filename {input.table} temp_{params.patient_id}_template.ipynb {output.notebook}
#         rm temp_{params.patient_id}.md
#         rm temp_{params.patient_id}_template.ipynb
#         """
