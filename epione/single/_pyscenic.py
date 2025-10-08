import os
import json
import zlib
import base64
import anndata as ad
#import pyscenic
import numpy as np
import pandas as pd
import loompy as lp

def pyscenic_grn(*,data_pathway,loom_pathway,download_pathway,adata):
  '''
  输入数据：
  data_pathway: 这就存中间文件的pathway (eg:/content)
  loom_pathway: 这就是存loom文件的pathway (eg:/content/pbmc10k_filtered_scenic.loom) 
  download_pathway: 这就是存下载文件的pathway (eg:/content)
  adata: 输入原始文件

  返回数据：
  TF调节子活性的aucell矩阵
  '''


  # 导入原始数据
  # # path to loom file with basic filtering applied (this will be created in the "initial filtering" step below). Optional.
  f_loom_path_scenic = loom_pathway
  # create basic row and column attributes for the loom file:
  row_attrs = {
    "Gene": np.array(adata.var_names) ,
  }
  col_attrs = {
    "CellID": np.array(adata.obs_names) ,
    "nGene": np.array( np.sum(adata.X.transpose()>0 , axis=0)).flatten() ,
    "nUMI": np.array( np.sum(adata.X.transpose() , axis=0)).flatten() ,
  }
  lp.create( f_loom_path_scenic, adata.X.transpose(), row_attrs, col_attrs)


  # grn步骤
  outpath_adj = os.path.join(data_pathway,'adj.csv')
  loom_path = loom_pathway
  tfs_path = os.path.join(download_pathway,'pySCENIC/resources/hs_hgnc_tfs.txt')  #下载的文件
  #!pyscenic grn {loom_path} {tfs_path} -o {outpath_adj} --num_workers 15

import os
import json
import zlib
import base64
import anndata as ad

def pyscenic_ctx_aucell(*,data_pathway,loom_pathway,download_pathway,save_pathway,adata):
  '''
  输入数据：
  data_pathway: 这就存中间文件的pathway
  download_pathway: 这就是存下载文件的pathway
  loom_pathway: 这就是存loom文件的pathway (eg:/content/pbmc10k_filtered_scenic.loom) 
  save_pathway: 这是存你保存好的aucell.loom文件的pathway
  adata: 输入原始文件

  返回数据：
  TF调节子活性的aucell矩阵
  '''

  # ctx步骤
  loom_path = loom_pathway
  motif_path = os.path.join(download_pathway,'motifs-v9-nr.hgnc-m0.001-o0.0.tbl')  #下载的文件
  db_path = os.path.join(download_pathway,'hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather') #下载的文件
  outpath_reg = os.path.join(data_pathway,'reg.csv')
  outpath_adj = os.path.join(data_pathway,'adj.csv')
  #pyscenic ctx {outpath_adj} \
  #      {db_path} \
  #      --annotations_fname {motif_path} \
  #      --expression_mtx_fname {loom_path} \
  #      --output {outpath_reg} \
  #      --mask_dropouts \
  #      --num_workers 15 > pyscenic_ctx_stdout.txt

  # aucell步骤
  #pyscenic aucell \
  #  {loom_path} \
  #  {outpath_reg} \
  #  --output {save_pathway} \
  #  --num_workers 15