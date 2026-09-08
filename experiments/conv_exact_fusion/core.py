"""Generate isolated same-operator variants; production remains untouched."""
from pathlib import Path
from experiments.conv_admission_band.core import Operator

HERE=Path(__file__).resolve().parent
BEGIN='            /* CONV_LEDGER_COST_BEGIN */\n'
END='            /* CONV_LEDGER_COST_END */\n'

def reference_source(source):
    """Reconstruct the pre-fusion baseline after production promotion."""
    if BEGIN in source:
        a=source.index(BEGIN);b=source.index(END,a)+len(END)
        source=source[:a]+(HERE/'original_ledger.c.inc').read_text()+source[b:]
    return source

def transform(source,projection=False,ledger=False,stream=False):
    source=reference_source(source)
    if projection:
        declaration='static void project_fibre(const float a[5], const int8_t sign[5], float delta, float c[5]) {'
        assert source.count(declaration)==1
        source=source.replace(declaration,declaration.replace('project_fibre(','project_fibre_original('))
        marker='typedef struct {\n    const float *x;\n    int n;\n    int lanes;\n    float *current;\n} projection_context;'
        source=source.replace(marker,(HERE/'fast_projection.h').read_text()+'\n'+marker)
    if ledger:
        begin=source.index('            double best_cost = INFINITY;',source.index('static void ordered_projection_worker'))
        end=source.index('            boundaries[boundary_count]',begin)
        old=source[begin:end]
        prefix='''            double prefix[11]={0},suffix[11]={0},left_cost[10],right_cost[10];
            const int count=local_end-local_begin+1;
            for(int index=0;index<count;++index){
                int slot=local_begin+index,cell=slot/5,k=slot%5;
                double value=current[(cell*5+k)*lanes+lane];
                double energy=value*value;
                left_cost[index]=coarse[knot-1]*value<0?energy:0;
                right_cost[index]=coarse[knot]*value<0?energy:0;
                prefix[index+1]=prefix[index]+left_cost[index];
            }
            for(int index=count-1;index>=0;--index)suffix[index]=suffix[index+1]+right_cost[index];
            double best_cost=INFINITY,second_cost=INFINITY;
            int best_boundary=begin;
            for(int candidate=begin;candidate<=end;++candidate){
                int index=candidate-local_begin;
                double cost=prefix[index]+suffix[index];
                if(cost<best_cost){second_cost=best_cost;best_cost=cost;best_boundary=candidate;}
                else if(cost<second_cost)second_cost=cost;
            }
            /* Preserve the legacy accumulation and first-minimum tie rule
               whenever roundoff might change the winner. */
            if(second_cost-best_cost<=64.0*DBL_EPSILON*(prefix[count]+suffix[0])){
'''
        if ledger=='rolling':
            prefix='''            double increments[10],cost=0.0,energy=0.0;
            for(int slot=local_begin;slot<=local_end;++slot){
                int cell=slot/5,k=slot%5;
                double value=current[(cell*5+k)*lanes+lane],square=value*value;
                int8_t expected=slot<begin?coarse[knot-1]:coarse[knot];
                if(expected*value<0)cost+=square;
                increments[slot-local_begin]=coarse[knot-1]*value<0?square:-square;
                energy+=square;
            }
            double best_cost=cost,second_cost=INFINITY;
            int best_boundary=begin;
            for(int candidate=begin+1;candidate<=end;++candidate){
                cost+=increments[candidate-1-local_begin];
                if(cost<best_cost){second_cost=best_cost;best_cost=cost;best_boundary=candidate;}
                else if(cost<second_cost)second_cost=cost;
            }
            /* At most ten squared samples and eight signed updates.
               An uncertain winner uses the original accumulation and ties. */
            if(!isfinite(energy) || second_cost-best_cost<=64.0*DBL_EPSILON*energy){
'''
        fallback=old.replace('            double best_cost = INFINITY;','            best_cost = INFINITY;').replace('            int best_boundary = begin;','            best_boundary = begin;')
        fallback=''.join('    '+line if line.strip() else line for line in fallback.splitlines(keepends=True))
        source=source[:begin]+BEGIN+prefix+fallback+'            }\n'+END+source[end:]
    if stream:
        source=source.replace('    int8_t *ledger = (int8_t *)malloc((size_t)slots);\n','')
        source=source.replace('if (!coarse || !ledger || !boundaries || !new_signs)','if (!coarse || !boundaries || !new_signs)')
        begin=source.index('        for (int slot = 0; slot < slots; ++slot) {',source.index('        int8_t active_sign = coarse[0];'))
        end=source.index('        for (int cell = 0; cell < cells; ++cell) {',begin)
        source=source[:begin]+source[end:]
        source=source.replace('                signs[k] = ledger[cell*5+k];',
'''                const int slot=cell*5+k;
                while(next_boundary<boundary_count && slot>=boundaries[next_boundary])
                    active_sign=new_signs[next_boundary++];
                signs[k]=active_sign;''')
        source=source.replace('free(new_signs); free(boundaries); free(ledger); free(coarse);','free(new_signs); free(boundaries); free(coarse);')
    return source

def operator(name):
    options={'base':{},'projection':dict(projection=True),'ledger':dict(ledger=True),
             'stream':dict(stream=True),'all':dict(projection=True,ledger=True,stream=True),
             'rolling':dict(ledger='rolling')}
    return Operator('conv',source_transform=lambda s:transform(s,**options[name]))
