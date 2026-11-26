"""Protocol adapter; algorithms and CLI defaults remain in upstream Go source."""
import argparse, concurrent.futures, csv, hashlib, io, json, os, pathlib, shutil, subprocess, tempfile, time, zipfile

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def execute(cmd, log, threads):
    start=time.monotonic()
    with open(log,'w') as f: subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,GOMAXPROCS=str(threads)))
    return time.monotonic()-start

def train(out):
    out=pathlib.Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    cpus=int(os.environ['BENCH_CPUS']);data=pathlib.Path(os.environ['BENCH_DATA_DIR'])/'train'
    manifest=json.loads((data/'manifest.json').read_text());runtime=pathlib.Path(os.environ['BENCH_RUNTIME_DIR'])
    groups=sorted(manifest['genomes'])
    if os.environ.get('BENCH_SMOKE')=='1':groups=[manifest['smoke_reference']]
    env=dict(os.environ,GOROOT=str(runtime/'go'),GOMODCACHE=str(runtime/'modules'),GOPATH=str(out/'gopath'),GOCACHE=str(out/'gocache'),GOPROXY='off',GOSUMDB='off',GOTOOLCHAIN='local',CGO_ENABLED='0',GOMAXPROCS=str(cpus))
    build_start=time.monotonic()
    subprocess.run([str(runtime/'go/bin/go'),'build','-p',str(cpus),'-mod=readonly','-trimpath','-tags','netgo','-ldflags=-s -w','-o',str(out/'lexicmap'),'./lexicmap'],check=True,env=env)
    source_digest=hashlib.sha256()
    for p in sorted(pathlib.Path('.').rglob('*')):
        if p.is_file() and '.git' not in p.parts and (p.suffix=='.go' or p.name in ('go.mod','go.sum')): source_digest.update(p.as_posix().encode()+p.read_bytes())
    resource={'cpus':cpus,'seed':int(os.environ['BENCH_SEED']),'binary_sha256':sha(out/'lexicmap'),'source_sha256':source_digest.hexdigest(),'compile_wall_s':time.monotonic()-build_start,'indexed_groups':groups,'index_bytes':0}
    (out/'idx').mkdir();start=time.monotonic()
    for acc in groups:
        g=manifest['genomes'][acc];ref=data/g['file'];assert sha(ref)==g['sha256']
        dest=out/'idx'/(acc+'.lmi')
        execute([out/'lexicmap','index',ref,'-j',cpus,'-O',dest],out/(acc+'.index.log'),cpus)
        resource['index_bytes']+=int(subprocess.check_output(['du','-sb',str(dest)],text=True).split()[0])
    resource['index_wall_s']=time.monotonic()-start
    (out/'resources.json').write_text(json.dumps(resource))
    print(json.dumps(resource),flush=True)

def predict(model,input_path,output):
    model=pathlib.Path(model).resolve();cpus=int(os.environ['BENCH_CPUS']);workers=max(1,cpus//8);threads=max(1,cpus//workers)
    resource=json.loads((model/'resources.json').read_text());assert resource['cpus']==cpus
    with zipfile.ZipFile(input_path) as z, tempfile.TemporaryDirectory(prefix='lexicmap-search-') as temp:
        tmp=pathlib.Path(temp);manifest=json.loads(z.read('manifest.json'));assert resource['indexed_groups']==manifest['indexed_groups']
        batches=manifest['batches']
        for b in batches:
            assert pathlib.Path(b['file']).name==b['file']
            (tmp/b['file']).write_bytes(z.read(b['file']))
        def search(b):
            dest=tmp/(b['name']+'.tsv')
            execute([model/'lexicmap','search','-d',model/'idx'/(b['reference']+'.lmi'),'-w',tmp/b['file'],'-j',threads,'-o',dest],tmp/(b['name']+'.log'),threads)
            with open(dest) as f:
                return [[r['query'],r['sseqid'],int(r['sstart']),int(r['send'])] for r in csv.DictReader(f,delimiter='\t')]
        start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool: outputs=list(pool.map(search,batches))
        resource['search_wall_s']=time.monotonic()-start
        resource['searched_batches']=[b['name'] for b in batches]
        result={'version':1,'input_sha256':sha(input_path),'query_ids':[q for b in batches for q in b['query_ids']],'hsps':[h for rows in outputs for h in rows],'resources':resource}
        pathlib.Path(output).write_text(json.dumps(result,separators=(',',':'))+'\n')
        print(json.dumps({'queries':len(result['query_ids']),'hsps':len(result['hsps']),'resources':resource}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    t=sub.add_parser('train');t.add_argument('--out',required=True)
    q=sub.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='train':train(a.out)
    else:predict(a.model,a.input,a.output)
