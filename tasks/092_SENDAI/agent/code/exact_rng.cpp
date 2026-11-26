#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <torch/library.h>
#include <mutex>
#include <cmath>
// Bulk MT19937 tempering: identical uint32 stream and double Bernoulli transform.
// The twist loops have minimum dependence distance 227 and safely auto-vectorize.
static void twist(at::mt19937_data_pod &d) {
    auto *s=d.state_.data();
    for(int i=0;i<227;i++) {uint32_t y=(s[i]&0x80000000u)|(s[i+1]&0x7fffffffu);s[i]=s[i+397]^(y>>1)^((s[i+1]&1u)*0x9908b0dfu);}
    for(int i=227;i<623;i++) {uint32_t y=(s[i]&0x80000000u)|(s[i+1]&0x7fffffffu);s[i]=s[i-227]^(y>>1)^((s[i+1]&1u)*0x9908b0dfu);}
    uint32_t y=(s[623]&0x80000000u)|(s[0]&0x7fffffffu);s[623]=s[396]^(y>>1)^((s[0]&1u)*0x9908b0dfu);
    d.left_=624;d.next_=0;
}
static uint32_t temper(uint32_t y) {
    y^=y>>11;y^=(y<<7)&0x9d2c5680u;y^=(y<<15)&0xefc60000u;y^=y>>18;return y;
}
at::Tensor mask(at::Tensor x,double p) {
    const uint64_t cutoff=static_cast<uint64_t>(std::ceil(std::ldexp(p,53)));
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.is_contiguous());
    auto result=at::empty_like(x);float *out=result.data_ptr<float>();auto n=result.numel();
    auto *gen=at::detail::getDefaultCPUGenerator().get<at::CPUGeneratorImpl>();
    std::lock_guard<std::mutex> lock(gen->mutex_);
    auto engine=gen->engine();auto d=engine.data();
    auto next=[&]() {if(--d.left_==0)twist(d);return temper(d.state_[d.next_++]);};
    int64_t i=0;
    while(i<n) {
        // At arbitrary odd offsets, one pair can straddle a twist boundary.
        if(d.left_<=2) {
            uint64_t a=next(), b=next();uint64_t r=(a<<32)|b;
            out[i++]=((r&((1ull<<53)-1)) < cutoff)?1.f:0.f;
            continue;
        }
        int64_t count=std::min<int64_t>(n-i,(d.left_-1)/2);
        auto *s=d.state_.data()+d.next_;
        for(int64_t j=0;j<count;j++) {
            uint64_t a=temper(s[2*j]),b=temper(s[2*j+1]);uint64_t r=(a<<32)|b;
            out[i+j]=((r&((1ull<<53)-1)) < cutoff)?1.f:0.f;
        }
        i+=count;d.next_+=2*count;d.left_-=2*count;
    }
    engine.set_data(d);gen->set_engine(engine);return result;
}
TORCH_LIBRARY(sendai_opt30,m) {m.def("mask(Tensor x, float p) -> Tensor");}
TORCH_LIBRARY_IMPL(sendai_opt30,CPU,m) {m.impl("mask",mask);}

// Return only selected spatial rows, but advance the RNG for every full-grid row.
at::Tensor selected_mask(at::Tensor x,int64_t points,at::Tensor indices,double p) {
    const uint64_t cutoff=static_cast<uint64_t>(std::ceil(std::ldexp(p,53)));
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.dim()==3);
    auto batch=x.size(0),k=x.size(1),width=x.size(2);
    auto result=at::empty_like(x);auto *out=result.data_ptr<float>();
    auto idx=indices.contiguous();auto *loc=idx.data_ptr<int64_t>();
    std::vector<std::pair<int64_t,int64_t>> sorted;
    for(int64_t j=0;j<k;j++)sorted.emplace_back(loc[j],j);
    std::sort(sorted.begin(),sorted.end());
    auto *gen=at::detail::getDefaultCPUGenerator().get<at::CPUGeneratorImpl>();
    std::lock_guard<std::mutex> lock(gen->mutex_);auto engine=gen->engine();auto d=engine.data();
    auto skip=[&](int64_t draws) {
        while(draws>0) {
            if(d.left_==1) {twist(d);d.left_=625;}
            int64_t take=std::min<int64_t>(draws,d.left_-1);
            d.left_-=take;d.next_+=take;draws-=take;
        }
    };
    auto next=[&]() {if(--d.left_==0)twist(d);return temper(d.state_[d.next_++]);};
    for(int64_t b=0;b<batch;b++) {
        int64_t position=0;
        for(auto [loc,j]:sorted) {
            skip(2*(loc-position)*width);
            for(int64_t w=0;w<width;w++) {
                uint64_t a=next(),v=next(),r=(a<<32)|v;
                out[(b*k+j)*width+w]=((r&((1ull<<53)-1))<cutoff)?1.f:0.f;
            }
            position=loc+1;
        }
        skip(2*(points-position)*width);
    }
    engine.set_data(d);gen->set_engine(engine);return result;
}
TORCH_LIBRARY_FRAGMENT(sendai_opt30,m) {m.def("selected_mask(Tensor x, int points, Tensor indices, float p) -> Tensor");}
TORCH_LIBRARY_IMPL(sendai_opt30,CPU,m) {m.impl("selected_mask",selected_mask);}
#include <ATen/Parallel.h>
// Compact saved masks and fused scaling reduce dropout memory traffic.
std::tuple<at::Tensor,at::Tensor> dropout_compact(at::Tensor x,double p,double scale) {
    const uint64_t cutoff=static_cast<uint64_t>(std::ceil(std::ldexp(p,53)));
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==at::kFloat && x.is_contiguous());
    auto mask=at::empty(x.sizes(),x.options().dtype(at::kByte));auto result=at::empty_like(x);
    auto n=x.numel();auto *bits=mask.data_ptr<uint8_t>();
    {
        auto *gen=at::detail::getDefaultCPUGenerator().get<at::CPUGeneratorImpl>();
        std::lock_guard<std::mutex> lock(gen->mutex_);auto engine=gen->engine();auto d=engine.data();
        auto next=[&]() {if(--d.left_==0)twist(d);return temper(d.state_[d.next_++]);};
        int64_t i=0;
        while(i<n) {
            if(d.left_<=2) {uint64_t a=next(),b=next(),r=(a<<32)|b;bits[i++]=((r&((1ull<<53)-1))<cutoff);continue;}
            int64_t count=std::min<int64_t>(n-i,(d.left_-1)/2);auto *s=d.state_.data()+d.next_;
            for(int64_t j=0;j<count;j++) {uint64_t a=temper(s[2*j]),b=temper(s[2*j+1]),r=(a<<32)|b;bits[i+j]=((r&((1ull<<53)-1))<cutoff);}
            i+=count;d.next_+=2*count;d.left_-=2*count;
        }
        engine.set_data(d);gen->set_engine(engine);
    }
    auto *src=x.data_ptr<float>();auto *dst=result.data_ptr<float>();float factor=scale;
    at::parallel_for(0,n,32768,[&](int64_t lo,int64_t hi){for(int64_t i=lo;i<hi;i++)dst[i]=src[i]*(bits[i]?factor:0.f);});
    return {result,mask};
}
at::Tensor dropout_compact_backward(at::Tensor grad,at::Tensor mask,double scale) {
    auto g=grad.contiguous();auto result=at::empty_like(g);auto n=g.numel();
    auto *src=g.data_ptr<float>();auto *dst=result.data_ptr<float>();auto *bits=mask.data_ptr<uint8_t>();float factor=scale;
    at::parallel_for(0,n,32768,[&](int64_t lo,int64_t hi){for(int64_t i=lo;i<hi;i++)dst[i]=src[i]*(bits[i]?factor:0.f);});return result;
}
TORCH_LIBRARY_FRAGMENT(sendai_opt30,m) {
    m.def("dropout_compact(Tensor x, float p, float scale) -> (Tensor, Tensor)");
    m.def("dropout_compact_backward(Tensor grad, Tensor mask, float scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(sendai_opt30,CPU,m) {
    m.impl("dropout_compact",dropout_compact);m.impl("dropout_compact_backward",dropout_compact_backward);
}
