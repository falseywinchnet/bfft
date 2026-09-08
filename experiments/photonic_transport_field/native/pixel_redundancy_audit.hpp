// Compile-time diagnostic only. No rounded key is used for value reuse.
struct PixelReuseEntry{RGB value{};std::uint64_t calls=0,computed=0;int bucket=-1;bool ready=false;};
using PixelReuseKey=std::array<std::uint64_t,12>;
struct PixelReuseAudit{
    int mode=0,bucket=0; // 0: census only, 1: reuse within bucket, 2: reuse across pixel.
    std::map<PixelReuseKey,PixelReuseEntry> entries;
    std::map<PixelReuseKey,std::uint64_t> geometry;
    std::map<PixelReuseKey,std::uint64_t> specular_geometry;
    std::uint64_t specular_calls=0;
    std::map<std::vector<int>,std::uint64_t> source_words;
    std::map<std::vector<int>,std::uint64_t> glossy_words;
    std::vector<int> previous_word;
    bool first_in_row=true;int glossy_depth=0;
    std::uint64_t source_intervals=0,glossy_intervals=0,adjacent_same_word=0,glossy_adjacent_same_word=0;
    std::uint64_t requests=0,computed=0,small_hits=0,wide_hits=0,inconsistent=0;
    PixelReuseKey key(int primitive,const Hit& hit,Vec3 view,bool directional,bool primary){
        PixelReuseKey k{};k[0]=primitive;k[1]=primary;k[2]=hit.front;
        const std::array<double,9> coordinates{hit.position.x,hit.position.y,hit.position.z,
            hit.normal.x,hit.normal.y,hit.normal.z,directional?view.x:0.,directional?view.y:0.,directional?view.z:0.};
        for(int i=0;i<9;++i)k[i+3]=std::bit_cast<std::uint64_t>(coordinates[i]==0?0.:coordinates[i]);
        return k;
    }
    PixelReuseEntry& request(int primitive,const Hit& hit,Vec3 view,bool directional,bool primary){
        ++requests;const auto k=key(primitive,hit,view,directional,primary);
        auto geom=k;geom[1]=0;geom[9]=geom[10]=geom[11]=0;++geometry[geom];
        auto& entry=entries[k];++entry.calls;return entry;
    }
    bool reuse(const PixelReuseEntry& entry)const{return entry.ready&&(mode==2||(mode==1&&entry.bucket==bucket));}
    void specular(int primitive,const Hit& hit){
        ++specular_calls;++specular_geometry[key(primitive,hit,{},false,false)];
    }
    void source_word(int receiver,int emitter,const SegmentProgram& program){
        if(mode!=0)return;
        std::vector<int> word{receiver,emitter};
        word.insert(word.end(),program.primitive.begin(),program.primitive.begin()+program.count);
        const bool repeated=!first_in_row&&word==previous_word;
        ++source_words[word];++source_intervals;adjacent_same_word+=repeated;
        if(glossy_depth){++glossy_words[word];++glossy_intervals;glossy_adjacent_same_word+=repeated;}
        previous_word=std::move(word);first_in_row=false;
    }
    void record(PixelReuseEntry& entry,const RGB& value,bool small_hit){
        if(entry.ready&&entry.value!=value)++inconsistent;
        entry.ready=true;entry.value=value;entry.bucket=bucket;
        if(small_hit)++small_hits;else{++computed;++entry.computed;}
    }
};
thread_local PixelReuseAudit* pixel_reuse_audit=nullptr;
struct PixelSpecularGuard{
    PixelReuseAudit* audit;
    explicit PixelSpecularGuard(PixelReuseAudit* value):audit(value){if(audit)++audit->glossy_depth;}
    ~PixelSpecularGuard(){if(audit)--audit->glossy_depth;}
};
