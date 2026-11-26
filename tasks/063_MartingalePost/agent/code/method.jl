include(joinpath(@__DIR__,"src","normal_biv.jl"))
input,output=ARGS
rows=readdlm(joinpath(input,"index.csv"),',',String;skipstart=1)
open(output,"w") do out
 println(out,"id,lo1,hi1,lo3,hi3,lo5,hi5")
 for row in eachrow(rows)
  id=row[1];n=parse(Int,row[2]);offset=parse(Int,row[3]);file=row[4]
  y=open(joinpath(input,file),"r") do io
   seek(io,offset);reshape(read!(io,Vector{Float64}(undef,2*n)),2,n)
  end
  mu_n=mean(y,dims=2);Sigma_n=cov(y,dims=2)
  theta=[mu_n[1],mu_n[2],Sigma_n[1,1],Sigma_n[2,2],Sigma_n[1,2]]
  B=2000;T_trunc=50
  theta_B_trunc=pr_sample_B(theta,n,T_trunc,B,382)
  theta_B_hybrid=copy(theta_B_trunc);add_tail(theta_B_hybrid,n,T_trunc)
  bounds=Float64[]
  for j in [1,3,5]
   sorted=sort(theta_B_hybrid[j,:]);push!(bounds,sorted[Int(B*.05/2)],sorted[Int(B*(1-.05/2))])
  end
  println(out,id,",",join(bounds,","))
 end
end
