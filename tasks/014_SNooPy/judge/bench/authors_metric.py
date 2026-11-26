"""Exact scoring function extracted from the existing authors evaluator."""
import os

def count_number_of_missing_variants(solution_calls_few_snps, solution_calls_many_snps, vcf_file):

    # Read VCF header to get contig lengths
    contig_lengths = {}
    with open(solution_calls_few_snps) as f:
        for line in f:
            if line.startswith("##contig="):
                # Example: ##contig=<ID=contig_12,length=15323>
                try:
                    id_part = line.split("ID=")[1]
                    contig_id = id_part.split(",")[0]
                    length_part = line.split("length=")[1]
                    contig_length = int(length_part.split(">")[0].replace(">", "").replace("]", "").replace("'", "").replace('"', '').strip())
                    contig_lengths[contig_id] = contig_length
                except Exception:
                    continue

    contigs_with_some_variants=set()

    #inventory the positions of the solution variants on the ref
    variant_pos_few = []
    with open(solution_calls_few_snps) as f:
        for line in f:
            if line.startswith("#"):
                continue
            ls = line.split()
            pos = int(ls[1]) - 1  # Convert 1-based to 0-based
            contigs_with_some_variants.add(ls[0])
            if any([len(i)==len(ls[3]) and i != "." for i in ls[4].split(',')]) :  #don't take into account big insertions or deletions, they are too hard to catch with just reads
                variant_pos_few.append((ls[0], pos,len(ls[3])))

    #convert variant_pos in one bool list per reference
    variant_presence_few = {c : [False] * contig_lengths[c] for c in contig_lengths.keys()}
    for contig, pos, length in variant_pos_few:
        for i in range(pos, pos + length):
            variant_presence_few[contig][i] = True

    # inventory the positions of the solution variants (many snps) on the ref
    variant_pos_many = []
    with open(solution_calls_many_snps) as f:
        for line in f:
            if line.startswith("#"):
                continue
            ls = line.split()
            pos = int(ls[1]) - 1  # Convert 1-based to 0-based
            if any([len(i) == len(ls[3]) and i != "." for i in ls[4].split(',')]):
                variant_pos_many.append((ls[0], pos, len(ls[3])))

    # convert variant_pos_many into one bool list per reference
    variant_presence_many = {c: [False] * contig_lengths[c] for c in contig_lengths.keys()}
    for contig, pos, length in variant_pos_many:
        for i in range(pos, pos + length):
            variant_presence_many[contig][i] = True

    # Accept a list of VCF files as input
    if isinstance(vcf_file, str):
        vcf_files = [vcf_file]
    else:
        vcf_files = vcf_file

    # Prepare output statistics for each VCF
    stats_per_vcf = []

    for vcf_path in vcf_files:
        # inventory the positions of the VCF variants on the ref
        vcf_variant_pos = []
        with open(vcf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                ls = line.split()
                pos = int(ls[1]) - 1  # Convert 1-based to 0-based
                if any([len(i)==len(ls[3]) and i != "." for i in ls[4].split(',')]) : 
                    vcf_variant_pos.append((ls[0], pos, len(ls[3])))

        # convert vcf_variant_pos into one bool list per reference
        vcf_variant_presence = {c: [False] * contig_lengths[c] for c in contig_lengths.keys()}
        for contig, pos, length in vcf_variant_pos:
            for i in range(pos, pos + length):
                vcf_variant_presence[contig][i] = True

        # Compare vcf_variant_presence and variant_presence to evaluate precision and recall
        missing_variants = []
        false_positives = []
        total_true_variants = 0
        total_variant_called = 0

        false_positive_vcf = f"{os.path.splitext(os.path.basename(vcf_path))[0]}_false_positive.vcf"
        missing_variant_vcf = f"{os.path.splitext(os.path.basename(vcf_path))[0]}_missing_variant.vcf"

        # # Clear previous output files if they exist
        open(false_positive_vcf, "w").close()
        # open(missing_variant_vcf, "w").close()

        for contig in contig_lengths:
            if contig not in contigs_with_some_variants:
                continue
            for i in range(contig_lengths[contig]):
                if variant_presence_few[contig][i]:
                    total_true_variants += 1
                    if not vcf_variant_presence[contig][i]:
                        missing_variants.append((contig, i))  
                        # with open(missing_variant_vcf, "a") as mvf:
                        #     mvf.write(f"{contig}\t{i+1}\t.\tN\tN\t.\t.\tMISSING\n")

                if vcf_variant_presence[contig][i]:
                    total_variant_called += 1
                    if not variant_presence_many[contig][i]:
                        false_positives.append((contig, i))
                        with open(false_positive_vcf, "a") as fpvf:
                            fpvf.write(f"{contig}\t{i+1}\t.\tN\tN\t.\t.\tFALSE_POSITIVE\n")

        precision = 1 - len(false_positives) / total_variant_called if total_variant_called > 0 else 0
        recall = 1 - len(missing_variants) / total_true_variants if total_true_variants > 0 else 0

        print(f"VCF: {vcf_path}")
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Number of true variants: ", total_true_variants, " and called ", total_variant_called)
        print("-" * 40)

        stats_per_vcf.append({
            "vcf": vcf_path,
            "precision": precision,
            "recall": recall,
            "true_variants": total_true_variants,
            "called_variants": total_variant_called,
            "missing_variants": missing_variants,
            "false_positives": false_positives,
            "missing_variant_vcf": missing_variant_vcf,
            "false_positive_vcf": false_positive_vcf
        })

    return stats_per_vcf
