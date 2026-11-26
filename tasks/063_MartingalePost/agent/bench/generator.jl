include(joinpath(@__DIR__,"generator_functions.jl"))
schedule=readdlm(ARGS[1],',',String;skipstart=1)
streams=Dict{String,IOStream}()
for row in eachrow(schedule)
 n=parse(Int,row[1]); seed=parse(Int,row[2]); path=row[3]
 io=get!(streams,path) do; open(path,"w"); end
 y=sim_data([-0.5,1.,1.,0.5,0.7],n,seed)
 write(io,y)
end
foreach(close,values(streams))
