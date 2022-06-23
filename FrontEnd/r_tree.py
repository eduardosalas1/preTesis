import numpy as np
from rtree import index
import face_recognition as fr
from timeit import default_timer as timer
 
def KNN_rtree(k, to_search, n):
# global start
# R-Tree Lecture from secondary memory
  
  path = "/home/elguille/mejoradoPRETESIS/BackEnd/Data/"

  rtree_name = path + 'rtreeFile'

  #rtree_name = path + 'rtree_' + str(n)
  #rtree_idx = process_collection(rtree_name, n)
  #print("R-tree generated")
# From image
  #query = encode_for_r(to_search)
  query=to_search
  p = index.Property()
  p.dimension = 128 #D
  p.buffering_capacity = 10 #M
  #p.dat_extension = 'data'
  #p.idx_extension = 'index'
  #idx = index.Index(rtree_name, properties=p)
  #rtreeidx = index.Rtree(rtree_name, properties=p)
  rtreeidx = index.Rtree(rtree_name, properties=p)  
  query_list = list(query)
  for query_i in query:
    query_list.append(query_i)

#  start = timer() 
  
  return rtreeidx.nearest(coordinates=query_list, num_results=k, objects='raw')
