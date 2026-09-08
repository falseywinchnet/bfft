struct RainDrop { float x=0,y=0,radius=1,speed=0,birth=0,phase=0;bool small=false; };
struct NormalBlob { float x,y,rx,ry,height; };
struct RainNormalTexture {
    int width,height,tile=32,cols,rows;
    std::vector<RainDrop> drops;
    std::vector<NormalBlob> blobs;
    std::vector<std::vector<int>> bins;
    std::vector<std::array<float,2>> mist;
    std::uint32_t random_state=0x7241494e;std::uint64_t merges=0,generation=0;
    float random(){random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return (random_state&0xffffff)/float(0x1000000);}
    explicit RainNormalTexture(int w,int h):width(w),height(h),cols((w+31)/32),rows((h+31)/32),bins(cols*rows),mist(256*256){
        const float scale=w/1280.f;
        for(int i=0;i<340;++i){RainDrop d;d.small=i>=70;d.x=random()*w;d.y=random()*h;
            d.radius=(d.small?1.8f+random()*3.2f:7.0f+random()*11.0f)*scale;
            d.speed=d.small?0:8+random()*20;d.birth=1.2f+random()*(d.small?9.0f:6.0f);d.phase=random()*6.283185f;drops.push_back(d);}
        // Periodic micro-droplet height field. Only its analytic derivatives
        // are retained; mist is a normal texture, never an opacity overlay.
        for(int i=0;i<6500;++i){const float x=random()*256,y=random()*256,r=.65f+random()*1.35f,h=.20f*r;
            const int lo_x=int(std::floor(x-r)),hi_x=int(std::ceil(x+r)),lo_y=int(std::floor(y-r)),hi_y=int(std::ceil(y+r));
            for(int yy=lo_y;yy<=hi_y;++yy)for(int xx=lo_x;xx<=hi_x;++xx){float dx=xx-x,dy=yy-y,q=(dx*dx+dy*dy)/(r*r);if(q>=1)continue;
                const float g=-4*h*(1-q)/(r*r);auto& n=mist[((yy+256)%256)*256+(xx+256)%256];n[0]+=g*dx;n[1]+=g*dy;}}
    }
    void update(float t,float dt){
        ++generation;const float scale=width/1280.f;
        for(auto& d:drops){if(t<d.birth)continue;
            if(!d.small){const float target=(14+3.8f*d.radius/scale)*scale;
                d.speed+=(target-d.speed)*std::min(1.f,dt*.7f);d.y+=d.speed*dt;
                d.x+=std::sin(d.phase+t*.8f)*2.5f*scale*dt;
            }else d.y+=.6f*scale*dt;
            if(d.y>height+d.radius*3){d.y=-d.radius*2;d.x=random()*width;d.birth=t+.15f+random()*.4f;}
        }
        // Sparse coalescence updates the simulation state, not scene geometry.
        for(std::size_t i=0;i<70;++i){auto& a=drops[i];if(t<a.birth+.3f)continue;
            for(std::size_t j=i+1;j<drops.size();++j){auto& b=drops[j];if(t<b.birth+.3f)continue;
                const float dx=a.x-b.x,dy=a.y-b.y,limit=.48f*(a.radius+b.radius);
                if(dx*dx+dy*dy>=limit*limit||a.radius>28*scale)continue;
                const float aa=a.radius*a.radius,bb=b.radius*b.radius;
                a.x=(a.x*aa+b.x*bb)/(aa+bb);a.y=(a.y*aa+b.y*bb)/(aa+bb);a.radius=std::sqrt(aa+bb);
                b.birth=t+2+random()*5;b.x=random()*width;b.y=-b.radius*2;++merges;
            }}
        blobs.clear();for(auto& b:bins)b.clear();
        auto add=[&](NormalBlob b){const int id=int(blobs.size());blobs.push_back(b);
            const int x0=std::max(0,int(std::floor((b.x-b.rx)/tile))),x1=std::min(cols-1,int(std::floor((b.x+b.rx)/tile)));
            const int y0=std::max(0,int(std::floor((b.y-b.ry)/tile))),y1=std::min(rows-1,int(std::floor((b.y+b.ry)/tile)));
            for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x)bins[y*cols+x].push_back(id);};
        for(const auto& d:drops){if(t<d.birth)continue;float growth=std::clamp((t-d.birth)/.55f,0.f,1.f);growth=growth*growth*(3-2*growth);
            const float r=d.radius*growth;if(r<.15f)continue;
            const float stretch=d.small?1.f:1.10f+std::min(d.speed/(100*scale),1.f)*.55f;
            add({d.x,d.y,r,r*stretch,r*(d.small?.27f:.34f)});
            if(!d.small&&d.speed>25*scale)add({d.x,d.y-r*1.7f,r*.30f,r*2.25f,r*.045f});
        }
    }
    std::array<float,2> normal(int x,int y,float t)const{
        float gx=0,gy=0;
        for(int id:bins[(y/tile)*cols+x/tile]){const auto& b=blobs[id];const float dx=x-b.x,dy=y-b.y;
            const float q=dx*dx/(b.rx*b.rx)+dy*dy/(b.ry*b.ry);if(q>=1)continue;
            const float h=-4*b.height*(1-q);gx+=h*dx/(b.rx*b.rx);gy+=h*dy/(b.ry*b.ry);
        }
        float amount=std::clamp((t-7.0f)/7.f,0.f,1.f);amount=amount*amount*(3-2*amount)*.34f;
        if(amount>0){const float xx=x*.85f+t*.9f,yy=y*.85f-t*.28f;const int ix=int(std::floor(xx)),iy=int(std::floor(yy));
            const float tx=xx-ix,ty=yy-iy;
            for(int j=0;j<2;++j)for(int i=0;i<2;++i){const auto& n=mist[((iy+j+1024)&255)*256+((ix+i)&255)];
                const float w=(i?tx:1-tx)*(j?ty:1-ty)*amount;gx+=n[0]*w;gy+=n[1]*w;}
        }
        // This bounded material-normal law is part of the simulator. The
        // response domain includes it completely; no query triggers a rebuild.
        const float scale=.55f/std::sqrt(.55f*.55f+gx*gx+gy*gy);
        return {gx*scale,-gy*scale};
    }
};
