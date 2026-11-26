# SNooPy
metagenomic SNP caller for long-read sequencing data. SNooPy relies on a statistical model tailored for metagenomics. At the moment where we ran our tests, it greatly outperformed other callers such as DeepVariant, Longshot and Bcftools on metagenomic datasets.

## Installation
Install SNooPy through conda:
```
conda create -n snoopy
conda activate snoopy
conda install bioconda::snoopy
```

You can also very esaily intall SNooPy manually.
SNooPy is a python file, which does not need compilation. You will need to have an environment with python3, scipy, numpy, pandas, pysam and sklearn installed. You can create such an environment in conda:
```
conda create -n snoopy_env -y -c bioconda -c conda-forge numpy scipy scikit-learn pysam pandas
conda activate snoopy_env
```
Then download the snoopy.py file from this repository. In the following command line, replace `snoopy` by `python /path/to/snoopy.py`

## Usage
```
snoopy --help

usage: snoopy [-h] -r REFERENCE -b BAM -o OUT [-t THREADS]
                     [--window WINDOW] [--benchmark]

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
                        Reference genome in FASTA format
  -b BAM, --bam BAM     Alignment file in BAM format
  -o OUT, --out-folder OUT
                        Name of the output folder
  -t THREADS, --threads THREADS
                        Number of threads to use [1]
  --window WINDOW       Size of window to perform read separation. Must be at
                        least twice shorter than average read length. 0 for
                        auto [0]
  --benchmark           Use if benchmarking to compare against other variant
                        callers [False]
```
### Input
SNooPy expects as input a reference genome (FASTA format) and a set of reads aligned to this reference in [BAM format](https://samtools.github.io/hts-specs/SAMv1.pdf), as provided e.g. by [minimap2](https://github.com/lh3/minimap2). You can obtain the BAM file by running, for example:
```
minimap2 -ax map-ont reference.fa reads.fq | samtools view -Sb | samtools sort -o alignment.bam
```

## Citation
SNooPy is published in NAR:

Faure, Roland & Faure, Ulysse & Truong, Tam & Derzelle, Alessandro & Lavenier, Dominique & Flot, Jean-François & Quince, Christopher. (2026). SNooPy: a statistical framework for long-read metagenomic variant calling. Nucleic Acids Research. 54. 10.1093/nar/gkag556. 
```
@article{article,
year = {2026},
month = {06},
pages = {},
author = {Faure, Roland and Faure, Ulysse and Truong, Tam and Derzelle, Alessandro and Lavenier, Dominique and Flot, Jean-François and Quince, Christopher},
title = {SNooPy: a statistical framework for long-read metagenomic variant calling},
volume = {54},
journal = {Nucleic Acids Research},
doi = {10.1093/nar/gkag556}
}
```
