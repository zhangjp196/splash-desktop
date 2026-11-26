// Independent CPU references for image-wide attention and the merger.
#pragma once
#include "metal/abi/Vision.h"
#include "ops/Inkling2VisionBlock.hpp"
#include <vector>
#include <limits>

static float inkBf(float x) { return test::bf16ToFloat(test::floatToBf16(x)); }
static void inkCompare(const char *name, const std::vector<float> &ref, const uint16_t *actual, float tolerance) {
  float maxError = 0, rms = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    float a = test::bf16ToFloat(actual[i]);
    test::require(std::isfinite(a), "non-finite GPU output");
    float e = std::abs(a - ref[i]); maxError = std::max(maxError, e); rms += e * e;
  }
  std::cout << name << " max_abs_error=" << maxError << " rms=" << std::sqrt(rms / ref.size()) << '\n';
  test::require(maxError <= tolerance, name);
}

static void inkAttention(metal::MetalBackend &backend, uint32_t tileWidth) {
  constexpr uint32_t H=1024, D=64, heads=16;
  const uint32_t W=2*tileWidth, T=1+4*tileWidth*2, P=(T+127)/128*128;
  auto bf = [&](size_t n) { return test::sharedBuffer(backend,n*2); };
  auto q=bf(T*H), k=bf(T*H), v=bf(T*H), qr=bf(P*H), kr=bf(P*H), vr=bf(P*H), a=bf(P*H), out=bf(T*H);
  auto *qp=static_cast<uint16_t*>(q.contents()), *kp=static_cast<uint16_t*>(k.contents()), *vp=static_cast<uint16_t*>(v.contents());
  for (uint32_t t=0;t<T;++t) for(uint32_t d=0;d<H;++d) {
    size_t i=t*H+d;
    qp[i]=test::floatToBf16(float(int((t*71+d*13+d/7)%97)-48)/32);
    kp[i]=test::floatToBf16(float(int((t*37+d*11+d/5)%89)-44)/32);
    vp[i]=test::floatToBf16(float(int((t*53+d*7+d/3)%83)-41)/16);
  }
  std::memset(qr.contents(),0x7f,P*H*2); std::memset(kr.contents(),0x7f,P*H*2); std::memset(vr.contents(),0x7f,P*H*2);
  metal::CommandGraph graph;
  graph.add("inkling2_vision_axial_qkv",{q,k,v,qr,kr,vr},Inkling2VisionRopeParams{T,P,H,heads,D,4,W,1,10000.0f},{P,1,1});
  graph.add("vision_attention_h64",{qr,kr,vr,a},VisionAttentionParams{T,P,0.125f},{(T+SPLASH_VISION_ATTENTION_ROWS-1)/SPLASH_VISION_ATTENTION_ROWS,heads,1},{SPLASH_VISION_ATTENTION_THREADS,1,1});
  graph.add("vision_attention_gather",{a,out},VisionQkvParams{T,P,H,D,D,heads},{T,1,1});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  std::vector<float> cq(T*H), ck(T*H), ref(T*H);
  auto rotate=[&](uint16_t *src, std::vector<float>& dst) {
    for(uint32_t t=0;t<T;++t) for(uint32_t h=0;h<heads;++h) for(uint32_t d=0;d<D;++d) {
      uint32_t i=t*H+h*D+d, pair=d/2, f=pair%16;
      float angle=t==0?0.0f:float((pair>=16?(t-1)/W:(t-1)%W)+1)*std::pow(10000.0f,-float(2*f)/32);
      dst[i]=inkBf(test::bf16ToFloat(src[i])*std::cos(angle)+(d%2?1:-1)*test::bf16ToFloat(src[i^1])*std::sin(angle));
    }
  };
  rotate(qp,cq); rotate(kp,ck);
  auto checkRope=[&](const char *name, const std::vector<float>& expected, metal::MetalBuffer &buffer) {
    std::vector<float> layout(P*H,0);
    for(uint32_t t=0;t<T;++t) for(uint32_t h=0;h<heads;++h) for(uint32_t d=0;d<D;++d) layout[(h*P+t)*D+d]=expected[t*H+h*D+d];
    inkCompare(name,layout,static_cast<uint16_t*>(buffer.contents()),0.016f);
  };
  checkRope("rope query incl CLS and padding",cq,qr); checkRope("rope key incl CLS and padding",ck,kr);
  std::vector<float> vl(P*H,0);
  for(uint32_t t=0;t<T;++t) for(uint32_t h=0;h<heads;++h) for(uint32_t d=0;d<D;++d) vl[(h*P+t)*D+d]=test::bf16ToFloat(vp[t*H+h*D+d]);
  inkCompare("value layout and padding",vl,static_cast<uint16_t*>(vr.contents()),0);
  for(uint32_t t=0;t<T;++t) for(uint32_t h=0;h<heads;++h) {
    std::vector<float> scores(T); float sum=0, mx=-std::numeric_limits<float>::infinity();
    for(uint32_t s=0;s<T;++s) { float dot=0; for(uint32_t d=0;d<D;++d) dot+=cq[t*H+h*D+d]*ck[s*H+h*D+d]; scores[s]=dot/8; mx=std::max(mx,scores[s]); }
    for(float &x:scores) { x=std::exp(x-mx); sum+=x; }
    for(uint32_t d=0;d<D;++d) { float x=0; for(uint32_t s=0;s<T;++s) x+=scores[s]/sum*test::bf16ToFloat(vp[s*H+h*D+d]); ref[t*H+h*D+d]=inkBf(x); }
  }
  std::cout << "attention tile grid 2x" << tileWidth << " tokens=" << T << '\n';
  inkCompare("image-wide attention",ref,static_cast<uint16_t*>(out.contents()),0.025f);
}

static void inkMerger(metal::MetalBackend &backend, uint32_t height, uint32_t width) {
  constexpr uint32_t H=1024, rows=32, K=4096, N=2304;
  const uint32_t tiles=height*width;
  auto bf=[&](size_t n){return test::sharedBuffer(backend,n*2);};
  auto hidden=test::sharedBuffer(backend,(tiles*4+1)*H*4), packed=bf(rows*K), up=bf(rows*K), down=bf(rows*N), w1=bf(K*K), w2=bf(N*K), zero=bf(K);
  auto *x=static_cast<float*>(hidden.contents());
  for(uint32_t i=0;i<(tiles*4+1)*H;++i) x[i]=float(int((i*31+i/1024*17)%127)-63)/32;
  std::memset(w1.contents(),0,K*K*2); std::memset(w2.contents(),0,N*K*2); std::memset(zero.contents(),0,K*2);
  auto *p1=static_cast<uint16_t*>(w1.contents()), *p2=static_cast<uint16_t*>(w2.contents());
  for(uint32_t d=0;d<K;++d) p1[d*K+(d*13+37)%K]=test::floatToBf16((d%2?-.75f:.5f));
  for(uint32_t d=0;d<N;++d) p2[d*K+(d*7+123)%K]=test::floatToBf16((d%3?-.5f:1.0f));
  metal::CommandGraph g;
  g.add("inkling2_vision_pack_merger",{hidden,packed},Inkling2VisionMergerParams{height,width,rows},{rows*16,1,1},{256,1,1});
  g.add("vision_gemm_m32n256_gelu_erf",{packed,w1,zero,up,up},VisionGemmParams{K,K},{1,16,1});
  g.add("vision_gemm_m32n256",{up,w2,zero,down,down},VisionGemmParams{N,K},{1,9,1});
  static_cast<void>(backend.submitCommandAsync(g.dispatches()).wait());
  std::vector<float> pr(rows*K,0), ur(rows*K,0), dr(rows*N,0);
  for(uint32_t r=0;r<tiles;++r) for(uint32_t c=0;c<K;++c) {
    uint32_t patch=c/H, dim=c%H, spatial=((r/width)*2+patch/2)*(2*width)+(r%width)*2+patch%2;
    pr[r*K+c]=inkBf(x[(spatial+1)*H+dim]);
  }
  for(uint32_t r=0;r<rows;++r) for(uint32_t d=0;d<K;++d) {float z=pr[r*K+(d*13+37)%K]*(d%2?-.75f:.5f); ur[r*K+d]=inkBf(.5f*z*(1+std::erf(z/std::sqrt(2.0f))));}
  for(uint32_t r=0;r<rows;++r) for(uint32_t d=0;d<N;++d) dr[r*N+d]=inkBf(ur[r*K+(d*7+123)%K]*(d%3?-.5f:1.0f));
  std::cout << "merger tile grid " << height << 'x' << width << '\n';
  inkCompare("merger packing and padding",pr,static_cast<uint16_t*>(packed.contents()),0);
  inkCompare("merger erf GELU",ur,static_cast<uint16_t*>(up.contents()),0.008f);
  inkCompare("merger output",dr,static_cast<uint16_t*>(down.contents()),0.008f);
}

// Sparse, non-diagonal matrices make a complete independent 23-block CPU
// reference inexpensive while still exercising every output and matrix layout.
static void inkBlocks(metal::MetalBackend &backend, uint32_t tileHeight, uint32_t tileWidth, uint32_t layers) {
  constexpr uint32_t H=1024, I=4096, P=128, R=64, D=64;
  const uint32_t T=1+4*tileHeight*tileWidth, W=2*tileWidth;
  auto bf=[&](size_t n){auto b=test::sharedBuffer(backend,n*2); std::memset(b.contents(),0,n*2); return b;};
  ops::Inkling2VisionBlockBuffers b;
  b.residual=test::sharedBuffer(backend,T*H*4);
  b.normalized=bf(R*H); b.query=bf(R*H); b.key=bf(R*H); b.value=bf(R*H);
  b.queries=bf(P*H); b.keys=bf(P*H); b.values=bf(P*H); b.attention=bf(P*H);
  b.context=bf(R*H); b.projected=bf(R*H); b.normalized2=bf(R*H); b.intermediate=bf(R*I); b.down=bf(R*H);
  auto vector=[&](uint32_t n,float base,float step){auto v=bf(n); auto p=static_cast<uint16_t*>(v.contents()); for(uint32_t i=0;i<n;++i) p[i]=test::floatToBf16(base+step*float(int(i%11)-5)); return v;};
  b.norm1Weight=vector(H,1,.0078125f); b.norm1Bias=vector(H,0,.00390625f);
  b.norm2Weight=vector(H,.9375f,.0078125f); b.norm2Bias=vector(H,0,.00390625f);
  b.queryBias=vector(H,0,.00390625f); b.keyBias=vector(H,0,-.00390625f);
  b.valueBias=vector(H,.015625f,.00390625f); b.outputBias=vector(H,0,.001953125f);
  b.upBias=vector(I,0,.00390625f); b.downBias=vector(H,0,.001953125f);
  b.scale1=vector(H,.125f,.001953125f); b.scale2=vector(H,.125f,-.001953125f);
  auto matrix=[&](uint32_t n,uint32_t k,uint32_t mul,uint32_t add,float val){auto w=bf(size_t(n)*k); auto p=static_cast<uint16_t*>(w.contents()); for(uint32_t d=0;d<n;++d) p[size_t(d)*k+(d*mul+add)%k]=test::floatToBf16(val*(d%2?-1:1)); return w;};
  b.queryWeight=matrix(H,H,7,13,.5f); b.keyWeight=matrix(H,H,11,19,.5f);
  b.valueWeight=matrix(H,H,13,23,.75f); b.outputWeight=matrix(H,H,17,29,.5f);
  b.upWeight=matrix(I,H,19,31,.75f); b.downWeight=matrix(H,I,23,37,.5f);
  auto get=[](const metal::MetalBuffer &v,uint32_t i){return test::bf16ToFloat(static_cast<const uint16_t*>(v.contents())[i]);};
  auto *actual=static_cast<float*>(b.residual.contents());
  std::vector<float> res(T*H), norm(T*H), q(T*H), k(T*H), v(T*H), context(T*H), temp(T*H), intermediate(T*I);
  for(uint32_t i=0;i<T*H;++i) actual[i]=res[i]=float(int((i*29+i/H*43)%251)-125)/128;
  auto normalize=[&](metal::MetalBuffer &w,metal::MetalBuffer &bias) {
    for(uint32_t t=0;t<T;++t) {double mean=0,var=0; for(uint32_t d=0;d<H;++d) mean+=res[t*H+d]; mean/=H;
      for(uint32_t d=0;d<H;++d) {double z=res[t*H+d]-mean; var+=z*z;} var/=H;
      for(uint32_t d=0;d<H;++d) norm[t*H+d]=inkBf(float((res[t*H+d]-mean)/std::sqrt(var+1e-5))*get(w,d)+get(bias,d)); }
  };
  auto project=[&](const std::vector<float>& in,std::vector<float>& out,uint32_t n,uint32_t kin,uint32_t mul,uint32_t add,float val,metal::MetalBuffer &bias,bool gelu) {
    for(uint32_t t=0;t<T;++t) for(uint32_t d=0;d<n;++d) {float z=in[t*kin+(d*mul+add)%kin]*val*(d%2?-1:1)+get(bias,d); if(gelu) z=.5f*z*(1+std::erf(z/std::sqrt(2.0f))); out[t*n+d]=inkBf(z);}
  };
  auto rotate=[&](std::vector<float>& a) { for(uint32_t t=1;t<T;++t) for(uint32_t h=0;h<16;++h) for(uint32_t pair=0;pair<32;++pair) {
    uint32_t i=t*H+h*D+pair*2; float angle=float((pair>=16?(t-1)/W:(t-1)%W)+1)*std::pow(10000.f,-float(2*(pair%16))/32);
    float x=a[i],y=a[i+1]; a[i]=inkBf(x*std::cos(angle)-y*std::sin(angle)); a[i+1]=inkBf(y*std::cos(angle)+x*std::sin(angle)); }};
  float overall=0;
  std::cout << "block tile grid " << tileHeight << 'x' << tileWidth << '\n';
  for(uint32_t layer=0;layer<layers;++layer) {
    metal::CommandGraph g; ops::Inkling2VisionBlock::encode(g,b,T,P,tileHeight,tileWidth,true);
    static_cast<void>(backend.submitCommandAsync(g.dispatches()).wait());
    normalize(b.norm1Weight,b.norm1Bias);
    project(norm,q,H,H,7,13,.5f,b.queryBias,false); project(norm,k,H,H,11,19,.5f,b.keyBias,false); project(norm,v,H,H,13,23,.75f,b.valueBias,false);
    rotate(q); rotate(k);
    for(uint32_t t=0;t<T;++t) for(uint32_t h=0;h<16;++h) {
      std::vector<float> scores(T); float sum=0,mx=-std::numeric_limits<float>::infinity();
      for(uint32_t s=0;s<T;++s) {float z=0; for(uint32_t d=0;d<D;++d) z+=q[t*H+h*D+d]*k[s*H+h*D+d]; scores[s]=z/8; mx=std::max(mx,scores[s]);}
      for(float &z:scores){z=std::exp(z-mx); sum+=z;}
      for(uint32_t d=0;d<D;++d){float z=0;for(uint32_t s=0;s<T;++s) z+=scores[s]/sum*v[s*H+h*D+d];context[t*H+h*D+d]=inkBf(z);}
    }
    project(context,temp,H,H,17,29,.5f,b.outputBias,false);
    for(uint32_t i=0;i<T*H;++i) res[i]+=temp[i]*get(b.scale1,i%H);
    normalize(b.norm2Weight,b.norm2Bias);
    project(norm,intermediate,I,H,19,31,.75f,b.upBias,true);
    project(intermediate,temp,H,I,23,37,.5f,b.downBias,false);
    float maxError=0; double sq=0;
    for(uint32_t i=0;i<T*H;++i) {res[i]+=temp[i]*get(b.scale2,i%H); test::require(std::isfinite(actual[i]),"non-finite block residual"); float e=std::abs(actual[i]-res[i]); maxError=std::max(maxError,e);sq+=double(e)*e;}
    overall=std::max(overall,maxError);
    if(layer==0 || layer+1==layers) std::cout << "block " << layer+1 << " max_abs_error=" << maxError << " rms=" << std::sqrt(sq/(T*H)) << '\n';
    test::require(maxError<.04f,"block CPU reference mismatch");
  }
  std::cout << layers << "-block maximum across layers=" << overall << '\n';
}
