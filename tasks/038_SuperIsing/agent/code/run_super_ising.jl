# Parameterised copy of code/examples/super_ising_spectrum.jl.
# Hamiltonian construction below is unchanged; cache terms/bases across parameters.
using FuzzifiED
using FuzzifiED.Fuzzifino
using LinearAlgebra, Printf, Random
Random.seed!(parse(Int, get(ENV, "BENCH_SEED", "0")))
FuzzifiED.ElementType = Float64
FuzzifiED.SilentStd = true
# The upstream example predates ObsMomIncr=true. Its eta*DPlus(eta)
# requires eta spin=-1/2; true silently gives a zero pair-conversion term.
# Select the library's original convention, preserving Eq.(7.12).
FuzzifiED.ObsMomIncr = false
# train.py runs one worker per lz2 sector; without ED_SECTOR, solve both here.
const SECTORS = haskey(ENV, "ED_SECTOR") ? [parse(Int, ENV["ED_SECTOR"])] : (0:1)
function build_model(nmf)
    t0 = time()
FuzzifiED.ObsNormRadSq = nmf
nof = 2 * nmf
nmb = nmf - 1
nob = nmb
qnd = [
    GetNeSQNDiag(nof, nob),
    GetBosonLz2SQNDiag(nof, nmb, 1) + SQNDiag(GetLz2QNDiag(nmf, 2), nob)
]
tms_l2 = GetL2STerms(nmf, 2, nmb, 1) 
cfs = Dict{Int64, SConfs}()
for lz = 0 : 1 
    cfs[lz] = SConfs(nof, nob, nmf, [nmf, lz], qnd)
end 

den_e = StoreComps(GetFerDensitySObs(nmf, 2) + GetBosDensitySObs(nmb, 1))
tms_int_e = SimplifyTerms(GetIntegral(den_e * den_e))

obs_bf = StoreComps(GetFermionSObs(nmf, 2, 1)' * GetBosonSObs(nmb, 1, 1))
tms_hop = SimplifyTerms(GetIntegral(obs_bf * DPlus(obs_bf))) ; 

den_σ = StoreComps(GetFerDensitySObs(nmf, 2, [0 1 ; 1 0]))
tms_int_σ = SimplifyTerms(GetIntegral(den_σ * Laplacian(den_σ))) 

den_χχ = StoreComps(GetBosDensitySObs(nmb, 1))
tms_int_yuk = SimplifyTerms(GetIntegral(den_σ * den_χχ + den_χχ * den_σ) / 2) 

tms_pol_σ = SimplifyTerms(GetIntegral(den_σ))
tms_pol_χχ= SimplifyTerms(GetIntegral(GetBosDensitySObs(nmb, 1)))
tms_pol_ϵ = SimplifyTerms(GetIntegral(GetFerDensitySObs(nmf, 2, [0 0 ; 0 1]))) 

filter!(tm -> length(tm.cstr) > 4, tms_int_e)
filter!(tm -> length(tm.cstr) > 4, tms_hop)
filter!(tm -> length(tm.cstr) > 4, tms_int_σ)
filter!(tm -> length(tm.cstr) > 4, tms_int_yuk)
@assert obs_bf.s2 == -1
@assert !isempty(tms_hop)
println("TERMS nmf=$nmf pair_spin2=$(obs_bf.s2) pair_conversion_terms=$(length(tms_hop))")
flush(stdout)

get_hmt_terms(t, U, g, hx, hz, μ) = SimplifyTerms(
    tms_int_e - t * (tms_hop + tms_hop')
    + U * tms_int_σ + g * tms_int_yuk
    - hx * tms_pol_σ - hz * tms_pol_ϵ - μ * tms_pol_χχ
) 


    bases = [SBasis(cfs[lz]) for lz in 0:1]
    l2_mats = [OpMat(SOperator(bs, tms_l2)) for bs in bases]
    nf_mats = [OpMat(SOperator(bs, STerms(GetPolTerms(nof, 2, [0 0; 0 1])) / nof)) for bs in bases]
    println("MODEL nmf=$nmf build_s=$(time()-t0) dimensions=$([bs.dim for bs in bases])")
    flush(stdout)
    return (get_hmt_terms, bases, l2_mats, nf_mats)
end

function run_super(model, nmf, hx, hz, mu, outpath)
    get_hmt_terms, bases, l2_mats, nf_mats = model
    t0 = time()
    tms_hmt = get_hmt_terms(1.5, 0.25, 1.0, hx, hz, mu)
    rows = []
    open(outpath * ".meta.tsv", "w") do meta
        println(meta, "lz2\tdim\tnnz\tmatrix_s\teigensystem_s\tobservables_s")
        for lz in SECTORS
            t1 = time()
            hmt_mat = OpMat(SOperator(bases[lz+1], tms_hmt))
            t2 = time()
            println("MATRIX nmf=$nmf lz2=$lz dim=$(hmt_mat.dimd) nnz=$(hmt_mat.nel) seconds=$(t2-t1)")
            flush(stdout)
            enrg, st = GetEigensystem(hmt_mat, 20)
            t3 = time()
            for i in eachindex(enrg)
                v = st[:,i]
                lv = real(dot(v, l2_mats[lz+1] * v))
                nv = real(dot(v, nf_mats[lz+1] * v))
                residual = norm(hmt_mat * v - enrg[i] * v)
                push!(rows, (enrg[i], lv, nv, lz, i, residual))
            end
            println(meta, join((lz, hmt_mat.dimd, hmt_mat.nel, t2-t1, t3-t2, time()-t3), '\t'))
            flush(meta)
            println("SECTOR nmf=$nmf lz2=$lz eigensystem_s=$(t3-t2) total_s=$(time()-t1)")
            flush(stdout)
            hmt_mat = nothing; st = nothing; GC.gc()
        end
    end
    sort!(rows, by=first)
    open(outpath, "w") do io
        println(io, "E,L2,nf,lz2,index,residual")
        for row in rows
            println(io, join(row, ','))
        end
    end
    println("DONE\t", outpath, "\t", time()-t0)
    flush(stdout)
end

function main()
    if length(ARGS) == 5
        nmf = parse(Int, ARGS[1])
        run_super(build_model(nmf), nmf, parse.(Float64, ARGS[2:4])..., ARGS[5])
        return
    end
    # Persistent foreground worker; each request is a complete ED calculation.
    model = nothing; cached_nmf = 0
    println("READY"); flush(stdout)
    for line in eachline(stdin)
        line == "QUIT" && break
        fields = split(line, '\t')
        nmf = parse(Int, fields[1])
        if nmf != cached_nmf
            model = nothing; GC.gc()
            model = build_model(nmf); cached_nmf = nmf
        end
        run_super(model, nmf, parse.(Float64, fields[2:4])..., fields[5])
    end
end
main()
