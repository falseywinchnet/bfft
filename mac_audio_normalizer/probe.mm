#import <Cocoa/Cocoa.h>
#include "VSTHost.hpp"
int main(int argc,char**argv) { @autoreleasepool {
    [NSApplication sharedApplication];
    GMax p(argv[1],48000);
    printf("inputs %d outputs %d parameters %d latency %d version %d\n",p.fx->numInputs,p.fx->numOutputs,p.fx->numParams,p.fx->initialDelay,p.fx->version);
    for(int i=0;i<p.fx->numParams;i++) {
       printf("%d %s default %g = %s %s\n",i,p.text(8,i).c_str(),p.fx->getParameter(p.fx,i),p.text(7,i).c_str(),p.text(6,i).c_str());
       for(float v: {0.f,.25f,.5f,.75f,1.f}) {p.fx->setParameter(p.fx,i,v);printf("  %.2f: %s\n",v,p.text(7,i).c_str());}
    }
} }
