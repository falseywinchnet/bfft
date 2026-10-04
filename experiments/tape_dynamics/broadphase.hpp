#pragma once
#include <algorithm>
#include <vector>
#include "math.hpp"

namespace tape {
using zc::phys::Vec3;
struct Bounds { Vec3 low,high; };
inline Bounds joined(Bounds a,Bounds b) {
    return {{std::min(a.low.x,b.low.x),std::min(a.low.y,b.low.y),std::min(a.low.z,b.low.z)},
            {std::max(a.high.x,b.high.x),std::max(a.high.y,b.high.y),std::max(a.high.z,b.high.z)}};
}
inline bool intersects(Bounds a,Bounds b) {
    return a.low.x<=b.high.x && b.low.x<=a.high.x && a.low.y<=b.high.y &&
           b.low.y<=a.high.y && a.low.z<=b.high.z && b.low.z<=a.high.z;
}
inline double area(Bounds a) {
    Vec3 d=a.high-a.low;return 2*(d.x*d.y+d.y*d.z+d.z*d.x);
}
// Spatial broad phase only. This stores swept geometric bounds; it does not
// schedule dynamics or determine tape order.
class Tree {
    struct Node { Bounds box{};int parent=-1,left=-1,right=-1,body=-1,height=0; };
    std::vector<Node> nodes;
    std::vector<int> free;
    int root=-1;
    int allocate() { if(free.empty()){nodes.emplace_back();return int(nodes.size()-1);}int n=free.back();free.pop_back();nodes[n]=Node{};return n; }
    void refit(int n) {
        while(n>=0){n=balance(n);Node& a=nodes[n];if(a.left>=0){a.box=joined(nodes[a.left].box,nodes[a.right].box);a.height=1+std::max(nodes[a.left].height,nodes[a.right].height);}n=a.parent;}
    }
    int balance(int i) {
        Node& a=nodes[i];if(a.left<0||a.height<2)return i;
        int b=a.left,c=a.right;int skew=nodes[c].height-nodes[b].height;
        if(skew>1){
            int f=nodes[c].left,g=nodes[c].right,old=a.parent;
            nodes[c].left=i;nodes[c].parent=old;a.parent=c;
            if(old<0)root=c;else if(nodes[old].left==i)nodes[old].left=c;else nodes[old].right=c;
            if(nodes[f].height>nodes[g].height){nodes[c].right=f;a.right=g;nodes[g].parent=i;nodes[f].parent=c;}
            else {nodes[c].right=g;a.right=f;nodes[f].parent=i;nodes[g].parent=c;}
            a.box=joined(nodes[a.left].box,nodes[a.right].box);a.height=1+std::max(nodes[a.left].height,nodes[a.right].height);
            nodes[c].box=joined(a.box,nodes[nodes[c].right].box);nodes[c].height=1+std::max(a.height,nodes[nodes[c].right].height);return c;
        }
        if(skew< -1){
            int d=nodes[b].left,e=nodes[b].right,old=a.parent;
            nodes[b].left=i;nodes[b].parent=old;a.parent=b;
            if(old<0)root=b;else if(nodes[old].left==i)nodes[old].left=b;else nodes[old].right=b;
            if(nodes[d].height>nodes[e].height){nodes[b].right=d;a.left=e;nodes[e].parent=i;nodes[d].parent=b;}
            else {nodes[b].right=e;a.left=d;nodes[d].parent=i;nodes[e].parent=b;}
            a.box=joined(nodes[a.left].box,nodes[a.right].box);a.height=1+std::max(nodes[a.left].height,nodes[a.right].height);
            nodes[b].box=joined(a.box,nodes[nodes[b].right].box);nodes[b].height=1+std::max(a.height,nodes[nodes[b].right].height);return b;
        }
        return i;
    }
public:
    int insert(int body,Bounds box) {
        int leaf=allocate();nodes[leaf].box=box;nodes[leaf].body=body;
        if(root<0){root=leaf;return leaf;}
        int sibling=root;
        while(nodes[sibling].left>=0){int l=nodes[sibling].left,r=nodes[sibling].right;
            double inherited=area(joined(nodes[sibling].box,box))-area(nodes[sibling].box);
            double lc=area(joined(nodes[l].box,box))+(nodes[l].left<0?0:-area(nodes[l].box))+inherited;
            double rc=area(joined(nodes[r].box,box))+(nodes[r].left<0?0:-area(nodes[r].box))+inherited;
            if(2*area(joined(nodes[sibling].box,box))<std::min(lc,rc))break;
            sibling=lc<rc?l:r;
        }
        int previous=nodes[sibling].parent,parent=allocate();nodes[parent].parent=previous;nodes[parent].box=joined(box,nodes[sibling].box);
        nodes[parent].left=sibling;nodes[parent].right=leaf;nodes[parent].height=nodes[sibling].height+1;
        nodes[sibling].parent=parent;nodes[leaf].parent=parent;
        if(previous<0)root=parent;else if(nodes[previous].left==sibling)nodes[previous].left=parent;else nodes[previous].right=parent;
        refit(parent);return leaf;
    }
    void erase(int leaf) {
        if(leaf==root){root=-1;free.push_back(leaf);return;}
        int parent=nodes[leaf].parent,grand=nodes[parent].parent,sibling=nodes[parent].left==leaf?nodes[parent].right:nodes[parent].left;
        if(grand<0){root=sibling;nodes[sibling].parent=-1;}
        else {if(nodes[grand].left==parent)nodes[grand].left=sibling;else nodes[grand].right=sibling;nodes[sibling].parent=grand;refit(grand);}
        free.push_back(parent);free.push_back(leaf);
    }
    template<class F> void query(Bounds box,F callback) const {
        if(root<0)return;std::vector<int> stack{root};
        while(!stack.empty()){int i=stack.back();stack.pop_back();const Node& n=nodes[i];if(!intersects(box,n.box))continue;
            if(n.left<0)callback(n.body);else {stack.push_back(n.right);stack.push_back(n.left);}}
    }
};
}
