import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm

from ._read import read_gtf
from .genome import Genome

def find_genes(adata,
                 genome,
                 key_added='gene_annotation',
                 upstream=5000,
                 downstream=0,
                 feature_type='gene',
                 annotation='HAVANA',
                 raw=False):
    """
    merge values of peaks/windows/features overlapping genebodies + 2kb upstream.
    It is possible to extend the search for closest gene to a given number of bases downstream as well.

    There is commonly 2 set of annotations in a gtf file(HAVANA, ENSEMBL). By default, the function
    will search annotation from HAVANA but other annotation label/source can be specifed.

    It is possible to use other type of features than genes present in a gtf file such as transcripts or CDS.

    """
    """
    Parameters
    ----------
    adata
        AnnData with peaks/features in `var_names` formatted like `chr_start_end`.
    genome
        A `Genome` object (recommended), or a path to a GTF/GFF annotation file (backward compatible).
    key_added
        Column name added to `adata.var` with gene annotations.
    upstream, downstream
        Flanking distances applied depending on strand when building gene windows.
    feature_type
        Feature type to match in annotation (e.g., 'gene', 'transcript').
    annotation
        Annotation source to match (e.g., 'HAVANA', 'ENSEMBL').
    raw
        Unused, kept for backward compatibility.
    """
    ### extracting the genes
    if isinstance(genome, Genome):
        anno_path = str(genome.annotation)
    else:
        anno_path = str(genome)

    gtf = {}
    with open(anno_path) as f:
        for line in f:
            if line[0:2] != '##' and '\t'+feature_type+'\t' in line and '\t'+annotation+'\t' in line:
                line = line.rstrip('\n').split('\t')
                if line[6] == '-':
                    if line[0] not in gtf.keys():
                        gtf[line[0]] = [[int(line[3])-downstream, int(line[4])+upstream,line[-1].split(';')[:-1]]]
                    else:
                        gtf[line[0]].append([int(line[3])-downstream, int(line[4])+upstream,line[-1].split(';')[:-1]])
                else:
                    if line[0] not in gtf.keys():
                        gtf[line[0]] = [[int(line[3])-upstream, int(line[4])+downstream,line[-1].split(';')[:-1]]]
                    else:
                        gtf[line[0]].append([int(line[3])-upstream, int(line[4])+downstream,line[-1].split(';')[:-1]])

    # extracting the feature coordinates
    raw_adata_features = {}
    feature_index = 0
    for line in adata.var_names.tolist():
        line = line.split('_')
        if line[0] not in raw_adata_features.keys():
            raw_adata_features[line[0]] = [[int(line[1]),int(line[2]), feature_index]]
        else:
            raw_adata_features[line[0]].append([int(line[1]),int(line[2]), feature_index])
        feature_index += 1

    ## find the genes overlaping the features.
    gene_index = []
    for chrom in raw_adata_features.keys():
        if chrom in gtf.keys():
            chrom_index = 0
            previous_features_index = 0
            for feature in raw_adata_features[chrom]:
                gene_name = []
                feature_start = feature[0]
                feature_end = feature[1]
                for gene in gtf[chrom]:
                    if (gene[1]<= feature_start): # the gene is before the feature. we need to test the next gene.
                        continue
                    elif (feature_end <= gene[0]): # the gene is after the feature. we need to test the next feature.
                        break
                    else: # the window is overlapping the gene.
                        for n in gene[-1]:
                            if 'gene_name' in n:
                                gene_name.append(n.lstrip('gene_name "').rstrip('""'))

                if gene_name == []:
                    gene_index.append('intergenic')
                elif len(gene_name)==1:
                    gene_index.append(gene_name[0])
                else:
                    gene_index.append(";".join(list(set(gene_name))))

        else:
            for feature in raw_adata_features[chrom]:
                gene_index.append("unassigned")

    adata.var[key_added] = gene_index


class Annotation(object):

    def __init__(self, genome_or_path) -> None:
        """Create an annotation helper from a Genome or annotation path."""
        gtf_path = genome_or_path.annotation if isinstance(genome_or_path, Genome) else genome_or_path
        self.gtf=read_gtf(str(gtf_path))
        self.features=self.gtf.loc[(self.gtf['feature']=='transcript')&(self.gtf['seqname'].str.contains('chr'))]
        chrom_dict={}
        for chrom in self.features['seqname'].unique():
            chrom_dict[chrom]=int(np.max(self.features.loc[self.features['seqname']==chrom].end))
        self.chrom_dict=chrom_dict

    def parse_peaks(self,chrom,peaks):
        peaks=int(peaks)
        if peaks<=0:
            return 0
        elif peaks>self.chrom_dict[chrom]:
            return self.chrom_dict[chrom]
        else:
            return peaks

    def tss_init(self,upstream=100,downstream=1000):
        features_pos=self.features.loc[self.features['strand']=='+']
        features_neg=self.features.loc[self.features['strand']=='-']

        features_pos['tss']=features_pos['start']
        features_pos['promoter']=["{}:{}-{}".format(i,self.parse_peaks(i,j-downstream),self.parse_peaks(i,j+upstream)) for i,j in zip(features_pos['seqname'],
                                                               features_pos['tss'])]
        
        features_neg['tss']=features_neg['end']
        features_neg['promoter']=["{}:{}-{}".format(i,self.parse_peaks(i,j-upstream),self.parse_peaks(i,j+downstream)) for i,j in zip(features_neg['seqname'],
                                                               features_neg['tss'])]
        self.features=pd.concat([features_pos,features_neg])

    def distal_init(self,upstream=[1000,200000],downstream=[1000,200000]):

        features_pos=self.features.loc[self.features['strand']=='+']
        features_neg=self.features.loc[self.features['strand']=='-']

        features_pos['up_distal']=["{}:{}-{}".format(i,self.parse_peaks(i,j-upstream[1]),self.parse_peaks(i,j-upstream[0])) for i,j in zip(features_pos['seqname'],
                                                               features_pos['tss'])]
        features_pos['down_distal']=["{}:{}-{}".format(i,self.parse_peaks(i,j+downstream[0]),self.parse_peaks(i,j+downstream[1])) for i,j in zip(features_pos['seqname'],
                                                               features_pos['tss'])]
        
        features_neg['down_distal']=["{}:{}-{}".format(i,self.parse_peaks(i,j-downstream[1]),self.parse_peaks(i,j-downstream[0])) for i,j in zip(features_neg['seqname'],
                                                               features_neg['tss'])]
        features_neg['up_distal']=["{}:{}-{}".format(i,self.parse_peaks(i,j+upstream[0]),self.parse_peaks(i,j+upstream[1])) for i,j in zip(features_neg['seqname'],
                                                               features_neg['tss'])]
        self.features=pd.concat([features_pos,features_neg])

    def body_init(self):

        features=self.features
        features['body']=["{}:{}-{}".format(i,j,k) for i,j,k in zip(features['seqname'],
                                                                features['start'],features['end'])]
        self.features=features

    # ---- Fast path implementations ----
    def _ensure_index(self, chrom: str):
        if not hasattr(self, '_iv_ix_fast'):
            self._iv_ix_fast = {}
        if chrom in self._iv_ix_fast:
            return
        sub = self.features.loc[self.features['seqname']==chrom]
        def parse_series(series):
            starts=[]; ends=[]; labels=[]
            for s in series.values.tolist():
                try:
                    part = s.split(":")[1]
                    a,b = part.split("-")
                    a=int(a); b=int(b)
                except Exception:
                    a=0; b=0
                starts.append(a); ends.append(b); labels.append(s)
            starts=np.asarray(starts, dtype=np.int64)
            ends=np.asarray(ends, dtype=np.int64)
            mids=(starts+ends)//2
            order=np.argsort(starts, kind='mergesort')
            return starts[order], ends[order], mids[order], np.asarray(labels, dtype=object)[order]
        def get_col(name):
            return sub[name] if name in sub.columns else pd.Series([], dtype=object)
        self._iv_ix_fast[chrom] = {
            'promoter': parse_series(get_col('promoter')),
            'up_distal': parse_series(get_col('up_distal')),
            'down_distal': parse_series(get_col('down_distal')),
            'body': parse_series(get_col('body')),
        }

    def query_fast(self, query_list, chrom='chr1'):
        """Vectorized overlap query using per-chromosome sorted interval indices.
        Returns a DataFrame with columns: promoter, promoter_overlap, updistal, updistal_overlap, downdistal,
        downdistal_overlap, body, body_overlap
        """
        self._ensure_index(chrom)
        iv = self._iv_ix_fast[chrom]
        q_intervals=[parse_interval(i) for i in query_list]
        out=pd.DataFrame(index=[f"{chrom}:{a}-{b}" for a,b in q_intervals],
                         columns=['promoter','promoter_overlap','updistal','updistal_overlap','downdistal','downdistal_overlap','body','body_overlap'])
        def pick(name, qs, qe, qmid, qlen):
            s,e,m,l = iv[name]
            if len(s)==0:
                return '', ''
            r = np.searchsorted(s, qe, side='right')
            if r==0:
                return '', ''
            s = s[:r]; e=e[:r]; m=m[:r]; l=l[:r]
            os = np.maximum(s, qs)
            oe = np.minimum(e, qe)
            mask = oe >= os
            if not np.any(mask):
                return '', ''
            olen = oe - os
            valid = mask & (olen > qlen*0.9)
            if not np.any(valid):
                return '', ''
            idx = np.argmin(np.abs(m[valid]-qmid))
            return l[valid][idx], f"{chrom}:{int(os[valid][idx])}-{int(oe[valid][idx])}"
        for (qs,qe) in tqdm(q_intervals):
            qlen = qe - qs
            qmid = qe - qlen//2
            p,po = pick('promoter', qs,qe,qmid,qlen)
            u,uo = pick('up_distal', qs,qe,qmid,qlen)
            d,do = pick('down_distal', qs,qe,qmid,qlen)
            b,bo = pick('body', qs,qe,qmid,qlen)
            out.loc[f"{chrom}:{qs}-{qe}"] = [p,po,u,uo,d,do,b,bo]
        out = out[~out.index.duplicated(keep='first')]
        return out

    def query_multi_fast(self, query_list, chrom='chr1', batch=1, ncpus=10):
        """Parallel fast querying by splitting the query list; each worker runs the fast picker.
        Note: building the index is done once in the parent process.
        """
        self._ensure_index(chrom)
        q_intervals=[parse_interval(i) for i in query_list]
        chunk_size = max(1, len(q_intervals)//max(1,batch))
        chunks = [q_intervals[i:i+chunk_size] for i in range(0, len(q_intervals), chunk_size)]
        def worker(chunk):
            iv = self._iv_ix_fast[chrom]
            out=pd.DataFrame(columns=['promoter','promoter_overlap','updistal','updistal_overlap','downdistal','downdistal_overlap','body','body_overlap'])
            def pick(name, qs, qe, qmid, qlen):
                s,e,m,l = iv[name]
                if len(s)==0:
                    return '', ''
                r = np.searchsorted(s, qe, side='right')
                if r==0:
                    return '', ''
                s2 = s[:r]; e2=e[:r]; m2=m[:r]; l2=l[:r]
                os = np.maximum(s2, qs)
                oe = np.minimum(e2, qe)
                mask = oe >= os
                if not np.any(mask):
                    return '', ''
                olen = oe - os
                valid = mask & (olen > qlen*0.9)
                if not np.any(valid):
                    return '', ''
                idx = np.argmin(np.abs(m2[valid]-qmid))
                return l2[valid][idx], f"{chrom}:{int(os[valid][idx])}-{int(oe[valid][idx])}"
            for qs,qe in chunk:
                qlen = qe-qs
                qmid = qe - qlen//2
                p,po = pick('promoter', qs,qe,qmid,qlen)
                u,uo = pick('up_distal', qs,qe,qmid,qlen)
                d,do = pick('down_distal', qs,qe,qmid,qlen)
                b,bo = pick('body', qs,qe,qmid,qlen)
                out.loc[f"{chrom}:{qs}-{qe}"] = [p,po,u,uo,d,do,b,bo]
            return out
        if ncpus and ncpus>1 and len(chunks)>1:
            import multiprocessing as _mp
            with _mp.Pool(processes=ncpus) as pool:
                dfs = pool.map(worker, chunks)
        else:
            dfs = [worker(c) for c in chunks]
        merge_pd = pd.concat(dfs)
        merge_pd = merge_pd[~merge_pd.index.duplicated(keep='first')]
        return merge_pd

    def query(self,query_list,chrom='chr1'):
        query_interval=[parse_interval(i) for i in query_list]
        promoter_interval=[parse_interval(i) for i in self.features.loc[self.features['seqname']==chrom,'promoter']]
        updistal_interval=[parse_interval(i) for i in self.features.loc[self.features['seqname']==chrom,'up_distal']]
        downdistal_interval=[parse_interval(i) for i in self.features.loc[self.features['seqname']==chrom,'down_distal']]
        body_interval=[parse_interval(i) for i in self.features.loc[self.features['seqname']==chrom,'body']]
        self.chrom=chrom
        self.promoter_interval=promoter_interval
        self.updistal_interval=updistal_interval
        self.downdistal_interval=downdistal_interval
        self.body_interval=body_interval

        merge_pd=pd.DataFrame(columns=['promoter','promoter_overlap',
                                       'updistal','updistal_overlap',
                                       'downdistal','downdistal_overlap',
                                       'body','body_overlap'])
        for interval1 in tqdm(query_interval):
            middle=interval1[1]-(interval1[1]-interval1[0])//2
            res_li=[]
            res_now_len=len(res_li)
            query_interval_len=interval1[1]-interval1[0]
            promoter_potential_interval={}
            promoter_overlap_potential_interval={}
            k=0
            for interval2 in promoter_interval:
                if k!=0:
                    if k<100:
                        k+=1
                    else:
                        break
                p_overlap_start = max(interval1[0], interval2[0])
                p_overlap_end = min(interval1[1], interval2[1])
                if p_overlap_start <= p_overlap_end:
                    if p_overlap_end-p_overlap_start>query_interval_len*0.9:
                        promoter_potential_interval[f"{chrom}:{interval2[0]}-{interval2[1]}"]=abs(interval2[1]-(interval2[1]-interval2[0])//2-middle)
                        promoter_overlap_potential_interval[f"{chrom}:{interval2[0]}-{interval2[1]}"]=f"{chrom}:{p_overlap_start}-{p_overlap_end}"
                        k+=1
                        #res_li.append(f"{chrom}:{interval2[0]}-{interval2[1]}")
                        #res_li.append(f"{chrom}:{p_overlap_start}-{p_overlap_end}")
                        #break
            if len(promoter_potential_interval)==0:
                res_li.append('')
                res_li.append('')
            else:
                min_key = min(promoter_potential_interval, key=promoter_potential_interval.get)
                res_li.append(min_key)
                res_li.append(promoter_overlap_potential_interval[min_key])
            res_now_len=len(res_li)

                    
            #if len(res_li)==res_now_len:
            #    res_li.append('')
            #    res_li.append('')
            #res_now_len=len(res_li)
            #print('promoter',res_now_len,res_li)
            updistal_potential_interval={}
            updistal_overlap_potential_interval={}
            k=0
            for interval2 in updistal_interval:
                if k!=0:
                    if k<100:
                        k+=1
                    else:
                        break
                u_overlap_start = max(interval1[0], interval2[0])
                u_overlap_end = min(interval1[1], interval2[1])
                if u_overlap_start <= u_overlap_end:
                    if u_overlap_end-u_overlap_start>query_interval_len*0.9:
                        updistal_potential_interval[f"{chrom}:{interval2[0]}-{interval2[1]}"]=abs(interval2[1]-(interval2[1]-interval2[0])//2-middle)
                        updistal_overlap_potential_interval[f"{chrom}:{interval2[0]}-{interval2[1]}"]=f"{chrom}:{u_overlap_start}-{u_overlap_end}"
                        k+=1
                        #res_li.append(f"{chrom}:{interval2[0]}-{interval2[1]}")
                        #res_li.append(f"{chrom}:{u_overlap_start}-{u_overlap_end}")
                        #break
            if len(updistal_potential_interval)==0:
                res_li.append('')
                res_li.append('')
            else:
                min_key = min(updistal_potential_interval, key=updistal_potential_interval.get)
                res_li.append(min_key)
                res_li.append(updistal_overlap_potential_interval[min_key])
            res_now_len=len(res_li)
            #if len(res_li)==res_now_len:
            #    res_li.append('')
            #    res_li.append('')
            #res_now_len=len(res_li)
            #print('updistal',res_now_len,res_li)
            downdistal_potential_interval={}
            downdistal_overlap_potential_interval={}
            k=0
            for interval2 in downdistal_interval:
                if k!=0:
                    if k<100:
                        k+=1
                    else:
                        break
                d_overlap_start = max(interval1[0], interval2[0])
                d_overlap_end = min(interval1[1], interval2[1])
                if d_overlap_start <= d_overlap_end:
                    if d_overlap_end-d_overlap_start>query_interval_len*0.9:
                        downdistal_potential_interval[f"{chrom}:{interval2[0]}-{interval2[1]}"]=abs(interval2[1]-(interval2[1]-interval2[0])//2-middle)
                        downdistal_overlap_potential_interval[f"{chrom}:{interval2[0]}-{interval2[1]}"]=f"{chrom}:{d_overlap_start}-{d_overlap_end}"
                        k+=1
                        #res_li.append(f"{chrom}:{interval2[0]}-{interval2[1]}")
                        #res_li.append(f"{chrom}:{d_overlap_start}-{d_overlap_end}")
                        #break
            if len(downdistal_potential_interval)==0:
                res_li.append('')
                res_li.append('')
            else:
                min_key = min(downdistal_potential_interval, key=downdistal_potential_interval.get)
                res_li.append(min_key)
                res_li.append(downdistal_overlap_potential_interval[min_key])
            res_now_len=len(res_li)
            #if len(res_li)==res_now_len:
            #    res_li.append('')
            #    res_li.append('')
            #res_now_len=len(res_li)
            body_potential_interval={}
            body_overlap_potential_interval={}
            k=0
            for interval2 in body_interval:
                if k!=0:
                    if k<100:
                        k+=1
                    else:
                        break
                b_overlap_start = max(interval1[0], interval2[0])
                b_overlap_end = min(interval1[1], interval2[1])
                if b_overlap_start <= b_overlap_end:
                    if b_overlap_end-b_overlap_start>query_interval_len*0.9:
                        body_potential_interval[f"{chrom}:{interval2[0]}-{interval2[1]}"]=abs(interval2[1]-(interval2[1]-interval2[0])//2-middle)
                        body_overlap_potential_interval[f"{chrom}:{interval2[0]}-{interval2[1]}"]=f"{chrom}:{b_overlap_start}-{b_overlap_end}"
                        #res_li.append(f"{chrom}:{interval2[0]}-{interval2[1]}")
                        #res_li.append(f"{chrom}:{b_overlap_start}-{b_overlap_end}")
                        #break
            if len(body_potential_interval)==0:
                res_li.append('')
                res_li.append('')
            else:
                min_key = min(body_potential_interval, key=body_potential_interval.get)
                res_li.append(min_key)
                res_li.append(body_overlap_potential_interval[min_key])
            #if len(res_li)==res_now_len:
            #    res_li.append('')
            #    res_li.append('')
            res_now_len=len(res_li)

            merge_pd.loc[f"{chrom}:{interval1[0]}-{interval1[1]}"]=res_li
        return merge_pd
    
    def process_query(self,query_interval):
        print('Start process_query{}...'.format(self.chrom))
        return self.process_overlap(query_interval)

    def process_overlap(self,query_interval):
        #print('process_overlap{}...'.format(self.chrom))
        merge_pd=pd.DataFrame(columns=['promoter','promoter_overlap',
                                       'updistal','updistal_overlap',
                                       'downdistal','downdistal_overlap',
                                       'body','body_overlap'])
        for interval1 in tqdm(query_interval):
            res_li=[]
            res_now_len=len(res_li)
            middle=interval1[1]-(interval1[1]-interval1[0])//2
            query_interval_len=interval1[1]-interval1[0]
            promoter_potential_interval={}
            promoter_overlap_potential_interval={}
            k=0
            for interval2 in self.promoter_interval:
                if k!=0:
                    if k<100:
                        k+=1
                    else:
                        break
                p_overlap_start = max(interval1[0], interval2[0])
                p_overlap_end = min(interval1[1], interval2[1])
                if p_overlap_start <= p_overlap_end:
                    if p_overlap_end-p_overlap_start>query_interval_len*0.9:
                        promoter_potential_interval[f"{self.chrom}:{interval2[0]}-{interval2[1]}"]=abs(interval2[1]-(interval2[1]-interval2[0])//2-middle)
                        promoter_overlap_potential_interval[f"{self.chrom}:{interval2[0]}-{interval2[1]}"]=f"{self.chrom}:{p_overlap_start}-{p_overlap_end}"
                        k+=1
                        #res_li.append(f"{self.chrom}:{interval2[0]}-{interval2[1]}")
                        #res_li.append(f"{self.chrom}:{p_overlap_start}-{p_overlap_end}")
                        #break
            if len(promoter_potential_interval)==0:
                res_li.append('')
                res_li.append('')
            else:
                min_key = min(promoter_potential_interval, key=promoter_potential_interval.get)
                res_li.append(min_key)
                res_li.append(promoter_overlap_potential_interval[min_key])
            res_now_len=len(res_li)
            #if len(res_li)==res_now_len:
            #    res_li.append('')
            #    res_li.append('')
            #res_now_len=len(res_li)
            #print('promoter',res_now_len,res_li)
            updistal_potential_interval={}
            updistal_overlap_potential_interval={}
            k=0
            for interval2 in self.updistal_interval:
                if k!=0:
                    if k<100:
                        k+=1
                    else:
                        break
                u_overlap_start = max(interval1[0], interval2[0])
                u_overlap_end = min(interval1[1], interval2[1])
                if u_overlap_start <= u_overlap_end:
                    if u_overlap_end-u_overlap_start>query_interval_len*0.9:
                        updistal_potential_interval[f"{self.chrom}:{interval2[0]}-{interval2[1]}"]=abs(interval2[1]-(interval2[1]-interval2[0])//2-middle)
                        updistal_overlap_potential_interval[f"{self.chrom}:{interval2[0]}-{interval2[1]}"]=f"{self.chrom}:{u_overlap_start}-{u_overlap_end}"
                        k+=1
                        #res_li.append(f"{self.chrom}:{interval2[0]}-{interval2[1]}")
                        #res_li.append(f"{self.chrom}:{u_overlap_start}-{u_overlap_end}")
                        #break
            if len(updistal_potential_interval)==0:
                res_li.append('')
                res_li.append('')
            else:
                min_key = min(updistal_potential_interval, key=updistal_potential_interval.get)
                res_li.append(min_key)
                res_li.append(updistal_overlap_potential_interval[min_key])
            #if len(res_li)==res_now_len:
            #    res_li.append('')
            #    res_li.append('')
            res_now_len=len(res_li)
            #print('updistal',res_now_len,res_li)
            downdistal_potential_interval={}
            downdistal_overlap_potential_interval={}
            k=0
            for interval2 in self.downdistal_interval:
                if k!=0:
                    if k<100:
                        k+=1
                    else:
                        break
                d_overlap_start = max(interval1[0], interval2[0])
                d_overlap_end = min(interval1[1], interval2[1])
                if d_overlap_start <= d_overlap_end:
                    if d_overlap_end-d_overlap_start>query_interval_len*0.9:
                        downdistal_potential_interval[f"{self.chrom}:{interval2[0]}-{interval2[1]}"]=abs(interval2[1]-(interval2[1]-interval2[0])//2-middle)
                        downdistal_overlap_potential_interval[f"{self.chrom}:{interval2[0]}-{interval2[1]}"]=f"{self.chrom}:{d_overlap_start}-{d_overlap_end}"
                        k+=1

                        #res_li.append(f"{self.chrom}:{interval2[0]}-{interval2[1]}")
                        #res_li.append(f"{self.chrom}:{d_overlap_start}-{d_overlap_end}")
                        #break
            if len(downdistal_potential_interval)==0:
                res_li.append('')
                res_li.append('')
            else:
                min_key = min(downdistal_potential_interval, key=downdistal_potential_interval.get)
                res_li.append(min_key)
                res_li.append(downdistal_overlap_potential_interval[min_key])
            #if len(res_li)==res_now_len:
            #    res_li.append('')
            #    res_li.append('')
            res_now_len=len(res_li)

            body_potential_interval={}
            body_overlap_potential_interval={}
            k=0
            for interval2 in self.body_interval:
                if k!=0:
                    if k<100:
                        k+=1
                    else:
                        break
                b_overlap_start = max(interval1[0], interval2[0])
                b_overlap_end = min(interval1[1], interval2[1])
                if b_overlap_start <= b_overlap_end:
                    if b_overlap_end-b_overlap_start>query_interval_len*0.9:
                        body_potential_interval[f"{self.chrom}:{interval2[0]}-{interval2[1]}"]=abs(interval2[1]-(interval2[1]-interval2[0])//2-middle)
                        body_overlap_potential_interval[f"{self.chrom}:{interval2[0]}-{interval2[1]}"]=f"{self.chrom}:{b_overlap_start}-{b_overlap_end}"
                        k+=1
                        #res_li.append(f"{self.chrom}:{interval2[0]}-{interval2[1]}")
                        #res_li.append(f"{self.chrom}:{b_overlap_start}-{b_overlap_end}")
                        #break
            if len(body_potential_interval)==0:
                res_li.append('')
                res_li.append('')
            else:
                min_key = min(body_potential_interval, key=body_potential_interval.get)
                res_li.append(min_key)
                res_li.append(body_overlap_potential_interval[min_key])
            #if len(res_li)==res_now_len:
            #    res_li.append('')
            #    res_li.append('')

            #print('downdistal',res_now_len,res_li)
            merge_pd.loc[f"{self.chrom}:{interval1[0]}-{interval1[1]}"]=res_li

        merge_pd=merge_pd[~merge_pd.index.duplicated(keep="first")]
        return merge_pd

    #multi process to merge using multiprocessing
    def query_multi(self,query_list,chrom='chr1',batch=1,ncpus=10):
        query_interval=[parse_interval(i) for i in query_list]
        promoter_interval=[parse_interval(i) for i in self.features.loc[self.features['seqname']==chrom,'promoter']]
        updistal_interval=[parse_interval(i) for i in self.features.loc[self.features['seqname']==chrom,'up_distal']]
        downdistal_interval=[parse_interval(i) for i in self.features.loc[self.features['seqname']==chrom,'down_distal']]
        body_interval=[parse_interval(i) for i in self.features.loc[self.features['seqname']==chrom,'body']]
        self.chrom=chrom
        self.promoter_interval=promoter_interval
        self.updistal_interval=updistal_interval
        self.downdistal_interval=downdistal_interval
        self.body_interval=body_interval

        merge_pd=pd.DataFrame(columns=['promoter','promoter_overlap',
                                       'updistal','updistal_overlap',
                                       'downdistal','downdistal_overlap',
                                       'body','body_overlap'])

        chunk_size = len(query_interval) // batch  # 计算每份的大小
        chunks = [query_interval[i:i+chunk_size] for i in range(0, len(query_interval), chunk_size)]

        with multiprocessing.Pool(processes=ncpus) as pool:
            results = pool.map(self.process_query, chunks)
        for result in results:
            merge_pd=pd.concat([merge_pd,result])

        return merge_pd
        
    def merge_info(self,merge_pd):
        merge_pd=merge_pd.fillna('')

        merge_pd['near_gene']=''
        merge_pd['peak_type']=''
        

        #promoter
        merge_pd['promoter_near_gene']=''
        merge_pd['promoter_range']=''
        merge_pd['promoter_overlap_range']=''
        merge_pd['promoter_min_range']=''
        merge_pd['promoter_vaild']=''
        merge_pd['promoter_near_gene_tss']=''


        promoter_li_pd=merge_pd.loc[merge_pd['promoter']!='']
        promoter_li_pd['promoter_range']=[int(i.split(':')[1].split('-')[1])-int(i.split(':')[1].split('-')[0]) for i in promoter_li_pd['promoter']]
        promoter_li_pd['promoter_overlap_range']=[int(i.split(':')[1].split('-')[1])-int(i.split(':')[1].split('-')[0]) for i in promoter_li_pd['promoter_overlap']]
        promoter_li_pd['promoter_min_range']=[min(i,j) for i,j in zip(promoter_li_pd['promoter_range'],promoter_li_pd['promoter_overlap_range'])]
        promoter_li_pd['promoter_vaild']=[i>=j*0.9 for i,j in zip(promoter_li_pd['promoter_overlap_range'],promoter_li_pd['promoter_min_range'])]

        promoter_li_index=promoter_li_pd.index
        test1=self.features.copy()
        test1.index=test1['promoter']
        test1=test1[~test1.index.duplicated(keep="first")]
        merge_pd.loc[promoter_li_index,'promoter_near_gene']=test1.loc[promoter_li_pd['promoter'].values.tolist(),'gene_name'].values.tolist()
        merge_pd.loc[promoter_li_index,'peak_type']='promoter'
        merge_pd.loc[promoter_li_index,'promoter_near_gene_tss']=test1.loc[promoter_li_pd['promoter'].values.tolist(),'tss'].values.tolist()

        merge_pd.loc[promoter_li_index,'promoter_range']=promoter_li_pd['promoter_range']
        merge_pd.loc[promoter_li_index,'promoter_overlap_range']=promoter_li_pd['promoter_overlap_range']
        merge_pd.loc[promoter_li_index,'promoter_min_range']=promoter_li_pd['promoter_min_range']
        merge_pd.loc[promoter_li_index,'promoter_vaild']=promoter_li_pd['promoter_vaild']

        #updistal
        merge_pd['updistal_near_gene']=''
        merge_pd['updistal_range']=''
        merge_pd['updistal_overlap_range']=''
        merge_pd['updistal_min_range']=''
        merge_pd['updistal_vaild']=''
        merge_pd['updistal_near_gene_tss']=''

        updistal_li_pd=merge_pd.loc[merge_pd['updistal']!='']
        updistal_li_pd['updistal_range']=[int(i.split(':')[1].split('-')[1])-int(i.split(':')[1].split('-')[0]) for i in updistal_li_pd['updistal']]
        updistal_li_pd['updistal_overlap_range']=[int(i.split(':')[1].split('-')[1])-int(i.split(':')[1].split('-')[0]) for i in updistal_li_pd['updistal_overlap']]
        updistal_li_pd['updistal_min_range']=[min(i,j) for i,j in zip(updistal_li_pd['updistal_range'],updistal_li_pd['updistal_overlap_range'])]
        updistal_li_pd['updistal_vaild']=[i>=j*0.9 for i,j in zip(updistal_li_pd['updistal_overlap_range'],updistal_li_pd['updistal_min_range'])]

        updistal_li_index=updistal_li_pd.index
        test1=self.features.copy()
        test1.index=test1['up_distal']
        test1=test1[~test1.index.duplicated(keep="first")]
        merge_pd.loc[updistal_li_index,'updistal_near_gene']=test1.loc[updistal_li_pd['updistal'].values.tolist(),'gene_name'].values.tolist()
        merge_pd.loc[updistal_li_index,'peak_type']='up_distal'
        merge_pd.loc[updistal_li_index,'updistal_near_gene_tss']=test1.loc[updistal_li_pd['updistal'].values.tolist(),'tss'].values.tolist()

        merge_pd.loc[updistal_li_index,'updistal_range']=updistal_li_pd['updistal_range']
        merge_pd.loc[updistal_li_index,'updistal_overlap_range']=updistal_li_pd['updistal_overlap_range']
        merge_pd.loc[updistal_li_index,'updistal_min_range']=updistal_li_pd['updistal_min_range']
        merge_pd.loc[updistal_li_index,'updistal_vaild']=updistal_li_pd['updistal_vaild']

        #down_distal
        merge_pd['downdistal_near_gene']=''
        merge_pd['downdistal_range']=''
        merge_pd['downdistal_overlap_range']=''
        merge_pd['downdistal_min_range']=''
        merge_pd['downdistal_vaild']=''
        merge_pd['downdistal_near_gene_tss']=''

        down_distal_li_pd=merge_pd.loc[merge_pd['downdistal']!='']
        down_distal_li_pd['downdistal_range']=[int(i.split(':')[1].split('-')[1])-int(i.split(':')[1].split('-')[0]) for i in down_distal_li_pd['downdistal']]
        down_distal_li_pd['downdistal_overlap_range']=[int(i.split(':')[1].split('-')[1])-int(i.split(':')[1].split('-')[0]) for i in down_distal_li_pd['downdistal_overlap']]
        down_distal_li_pd['downdistal_min_range']=[min(i,j) for i,j in zip(down_distal_li_pd['downdistal_range'],down_distal_li_pd['downdistal_overlap_range'])]
        down_distal_li_pd['downdistal_vaild']=[i>=j*0.9 for i,j in zip(down_distal_li_pd['downdistal_overlap_range'],down_distal_li_pd['downdistal_min_range'])]

        down_distal_li_index=down_distal_li_pd.index
        test1=self.features.copy()
        test1.index=test1['down_distal']
        test1=test1[~test1.index.duplicated(keep="first")]
        merge_pd.loc[down_distal_li_index,'downdistal_near_gene']=test1.loc[down_distal_li_pd['downdistal'].values.tolist(),'gene_name'].values.tolist()
        merge_pd.loc[down_distal_li_index,'peak_type']='down_distal'
        merge_pd.loc[down_distal_li_index,'downdistal_near_gene_tss']=test1.loc[down_distal_li_pd['downdistal'].values.tolist(),'tss'].values.tolist()

        merge_pd.loc[down_distal_li_index,'downdistal_range']=down_distal_li_pd['downdistal_range']
        merge_pd.loc[down_distal_li_index,'downdistal_overlap_range']=down_distal_li_pd['downdistal_overlap_range']
        merge_pd.loc[down_distal_li_index,'downdistal_min_range']=down_distal_li_pd['downdistal_min_range']
        merge_pd.loc[down_distal_li_index,'downdistal_vaild']=down_distal_li_pd['downdistal_vaild']

        #body
        merge_pd['body_near_gene']=''
        merge_pd['body_range']=''
        merge_pd['body_overlap_range']=''
        merge_pd['body_min_range']=''
        merge_pd['body_vaild']=''
        merge_pd['body_near_gene_tss']=''

        body_li_pd=merge_pd.loc[merge_pd['body']!='']
        body_li_pd['body_range']=[int(i.split(':')[1].split('-')[1])-int(i.split(':')[1].split('-')[0]) for i in body_li_pd['body']]
        body_li_pd['body_overlap_range']=[int(i.split(':')[1].split('-')[1])-int(i.split(':')[1].split('-')[0]) for i in body_li_pd['body_overlap']]
        body_li_pd['body_min_range']=[min(i,j) for i,j in zip(body_li_pd['body_range'],body_li_pd['body_overlap_range'])]
        body_li_pd['body_vaild']=[i>=j*0.9 for i,j in zip(body_li_pd['body_overlap_range'],body_li_pd['body_min_range'])]

        body_li_index=body_li_pd.index
        test1=self.features.copy()
        test1.index=test1['body']
        test1=test1[~test1.index.duplicated(keep="first")]

        merge_pd.loc[body_li_index,'body_near_gene']=test1.loc[body_li_pd['body'].values.tolist(),'gene_name'].values.tolist()
        merge_pd.loc[body_li_index,'peak_type']='body'
        merge_pd.loc[body_li_index,'body_near_gene_tss']=test1.loc[body_li_pd['body'].values.tolist(),'tss'].values.tolist()

        merge_pd.loc[body_li_index,'body_range']=body_li_pd['body_range']
        merge_pd.loc[body_li_index,'body_overlap_range']=body_li_pd['body_overlap_range']
        merge_pd.loc[body_li_index,'body_min_range']=body_li_pd['body_min_range']
        merge_pd.loc[body_li_index,'body_vaild']=body_li_pd['body_vaild']

        merge_pd['peaktype']='intergenic'
        merge_pd.loc[merge_pd['updistal_vaild']==True,'peaktype']='up_distal'
        merge_pd.loc[merge_pd['downdistal_vaild']==True,'peaktype']='down_distal'
        merge_pd.loc[merge_pd['body_vaild']==True,'peaktype']='body'
        merge_pd.loc[merge_pd['promoter_vaild']==True,'peaktype']='promoter'

        merge_pd['neargene']=''
        merge_pd.loc[merge_pd['peaktype']=='up_distal','neargene']=merge_pd.loc[merge_pd['peaktype']=='up_distal','updistal_near_gene'].values.tolist()
        merge_pd.loc[merge_pd['peaktype']=='down_distal','neargene']=merge_pd.loc[merge_pd['peaktype']=='down_distal','downdistal_near_gene'].values.tolist()
        merge_pd.loc[merge_pd['peaktype']=='body','neargene']=merge_pd.loc[merge_pd['peaktype']=='body','body_near_gene'].values.tolist()
        merge_pd.loc[merge_pd['peaktype']=='promoter','neargene']=merge_pd.loc[merge_pd['peaktype']=='promoter','promoter_near_gene'].values.tolist()

        merge_pd['neargene_tss']=''
        merge_pd.loc[merge_pd['peaktype']=='up_distal','neargene_tss']=merge_pd.loc[merge_pd['peaktype']=='up_distal','updistal_near_gene_tss'].values.tolist()
        merge_pd.loc[merge_pd['peaktype']=='down_distal','neargene_tss']=merge_pd.loc[merge_pd['peaktype']=='down_distal','downdistal_near_gene_tss'].values.tolist()
        merge_pd.loc[merge_pd['peaktype']=='body','neargene_tss']=merge_pd.loc[merge_pd['peaktype']=='body','body_near_gene_tss'].values.tolist()
        merge_pd.loc[merge_pd['peaktype']=='promoter','neargene_tss']=merge_pd.loc[merge_pd['peaktype']=='promoter','promoter_near_gene_tss'].values.tolist()

        return merge_pd

    def add_gene_info(self,adata,merge_pd,
                        columns=['peaktype','neargene','neargene_tss']):
        adata.var=adata.var.assign(**merge_pd[columns])

        

class easter_egg(object):

    def __init__(self,):
        print('Easter egg is ready to be hatched!')

    def O(self):
        print('尊嘟假嘟')
    

# 解析区间列表为起始和结束值的元组
def parse_interval(interval_str):
    parts = interval_str.split(":")[1].split("-")
    start = int(parts[0])
    end = int(parts[1])
    return start, end
