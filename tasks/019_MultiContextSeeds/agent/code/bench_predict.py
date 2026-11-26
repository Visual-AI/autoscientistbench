"""Protocol wrapper: run the freshly built mapper and serialize primary intervals."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    model,inp,out=Path(args.model),Path(args.input),Path(args.output)
    cfg=json.loads((inp/'input.json').read_text())
    cpus=int(os.environ.get('BENCH_CPUS','12'))
    seed=int(os.environ.get('BENCH_SEED','1'))
    ref=inp/'reference.fa' if cfg['reference']=='local' else Path(os.environ['BENCH_DATA_DIR'])/'train/reference.fa'
    # Default extension alignment with MCS enabled, exactly as the reproduction.
    command=[str(model/'strobealign'),'-v','-t',str(cpus),str(ref),str(inp/'reads.fastq.gz')]
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bio19-') as tmp:
        sam=Path(tmp)/'primary.sam'
        log=Path(tmp)/'mapper.log'
        with sam.open('w') as stdout,log.open('w') as stderr:
            subprocess.run(command,stdout=stdout,stderr=stderr,check=True)
        diagnostic=log.read_text()
        print(diagnostic,end='')
        times=re.findall(r'Total time mapping:\s*([0-9.eE+\-]+)\s+s\.',diagnostic)
        if len(times)!=1: raise ValueError('missing or ambiguous mapper runtime')
        meta={'schema':1,'mapping_time_s_tool_reported':float(times[0]),'threads':cpus,'seed':seed}
        with out.open('w') as target,sam.open() as source:
            target.write(json.dumps(meta,allow_nan=False)+'\n')
            target.write('id\treference\tstart\tend\n')
            for line in source:
                if line.startswith('@'):continue
                row=line.rstrip('\n').split('\t')
                flag=int(row[1])
                if flag & (256|2048):continue
                name=row[0]
                if flag & 4: chrom,start,end='*',-1,-1
                else:
                    chrom,start=row[2],int(row[3])-1
                    cigar=re.findall(r'(\d+)([MIDNSHP=X])',row[5])
                    if ''.join(n+op for n,op in cigar)!=row[5]: raise ValueError('invalid CIGAR')
                    end=start+sum(int(n) for n,op in cigar if op in 'MDN=X')
                target.write(f'{name}\t{chrom}\t{start}\t{end}\n')

if __name__=='__main__':main()
