/* Freestanding, persistent-atlas SIMD warp. Double precision is retained.
 * Native source admission runs once; only the inverse map changes.
 * No intermediate raster is constructed. No relaxed SIMD / fast-math.
 */
#include <wasm_simd128.h>
#include <math.h>
typedef unsigned char u8;
typedef unsigned int u32;
#define EXPORT(name) __attribute__((export_name(name)))
extern u8 __heap_base;
static u32 cursor;
static double *control,*source,*matrix,*quad,*metrics,*probe;
static u8 *output,*comparison;
static int sw,sh,lw,native_mode;
static u32 source_end,output_capacity,maximum_pixels;
static double min(double a,double b){return a<b?a:b;}
static double max(double a,double b){return a>b?a:b;}
static double clamp(double x,double lo,double hi){return min(hi,max(lo,x));}
static u32 reserve(u32 bytes){
  u32 p=(cursor+15u)&~15u, end=p+bytes;
  if(end<p)return 0;
  unsigned long long size=(unsigned long long)__builtin_wasm_memory_size(0)*65536u;
  if(end>size&&__builtin_wasm_memory_grow(0,(end-size+65535u)/65536u)==(u32)-1)return 0;
  cursor=end;return p;
}
EXPORT("prepare") int prepare(int w,int h,int limit){
  if(w<5||h<5||w>129||h>129||limit<25||limit>16777216)return 1;
  native_mode=0;cursor=(u32)&__heap_base;sw=w;sh=h;lw=5*(w-1)+1;maximum_pixels=limit;output_capacity=0;output=comparison=0;
  control=(double*)reserve((u32)lw*(5*(h-1)+1)*4*8);
  source=(double*)reserve((u32)w*h*4*8);
  matrix=(double*)reserve(9*8);quad=(double*)reserve(8*8);
  metrics=(double*)reserve(4*8);probe=(double*)reserve(4*8);
  source_end=cursor;
  return !control||!source||!matrix||!quad||!metrics||!probe;
}
EXPORT("control_pointer") u32 control_pointer(){return (u32)control;}
EXPORT("source_pointer") u32 source_pointer(){return (u32)source;}
EXPORT("matrix_pointer") u32 matrix_pointer(){return (u32)matrix;}
EXPORT("quad_pointer") u32 quad_pointer(){return (u32)quad;}
EXPORT("metrics_pointer") u32 metrics_pointer(){return (u32)metrics;}
EXPORT("output_pointer") u32 output_pointer(){return (u32)output;}
EXPORT("comparison_pointer") u32 comparison_pointer(){return (u32)comparison;}
static void bernstein(double t,double *b){
  double s=1-t,tt=t*t,ss=s*s,t3=tt*t,s3=ss*s;
  b[0]=ss*s3;b[1]=5*t*ss*ss;b[2]=10*tt*s3;b[3]=10*t3*ss;b[4]=5*tt*tt*s;b[5]=tt*t3;
}
