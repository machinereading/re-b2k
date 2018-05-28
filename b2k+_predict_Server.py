# /usr/bin/python
# -*- coding: utf-8 -*-

# korean encoding
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
import numpy
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from sklearn.linear_model import LogisticRegression
from collections import namedtuple
from bottle import run, post, request, response

import datetime
print '[START]', sys.argv[0], datetime.datetime.now()
# ======================================================================
# input configurations
# b2k model
b2k_sent_model_filename   = 'model/b2k_plus_sent_model'
classifier_model_filename = 'model/b2k_plus_classifier'

kb_file                   = 'model/mappingbased_objects_ko.ttl'

sep = ':-:'
# ======================================================================

print "\tLoading trained model...", datetime.datetime.now()
from gensim.models import Doc2Vec
b2k_sent_model   = Doc2Vec.load(b2k_sent_model_filename)

# Common function to process data structures 
def buildListWithSet1(givenlist, key, value):
  try:
    givenlist[key] = givenlist[key] | set([value])
  except KeyError:
    givenlist[key] = set([value])

def buildListWithSet(givenlist, key, valueSet):
  try:
    givenlist[key] = givenlist[key] | valueSet
  except KeyError:
    givenlist[key] = valueSet

def find_entity_term(sentence):
  import re
  entity_tag = re.compile(".*?\[(.*?)\]")
  entity_terms = set([])
  linktext_in_sentence = re.findall(entity_tag, sentence)

  for x in linktext_in_sentence:    
    entity_terms.add(x)

  return entity_terms

import itertools
def findsubset(S,m):
  return set(itertools.combinations(S,m))


print "\tReading KB for entity-graph construction...", datetime.datetime.now()
kb_memory = {}
p_sbj_info = {}
p_obj_info = {}
for row in open(kb_file):
  s = row.strip().split(' ')[0].replace('<http://ko.dbpedia.org/resource/','').replace('>','')
  o = row.strip().split(' ')[2].replace('<http://ko.dbpedia.org/resource/','').replace('>','')
  p = row.strip().split(' ')[1].replace('<http://dbpedia.org/ontology/','').replace('>','')
  buildListWithSet1(kb_memory, s+sep+o, p)
  buildListWithSet1(p_sbj_info, s, p)
  buildListWithSet1(p_obj_info, o, p)


def process_b2k_extraction(b2k_input_doc, b2k_output):
  print "\tAnalyzing input text...", datetime.datetime.now()
  import collections
  ent_by_paragraph = collections.OrderedDict()
  sen_by_paragraph = {}
  ent_by_sen = {}
  paragraph_idx = 0
  for row in open(b2k_input_doc, 'r'):
    sentence = row.strip()
    # print sentence
    entity_set = set([])
    # separating paragraphs from the given input document
    if sentence == '':      # new line with no characters
      paragraph_idx =+ 1
    else:

      entity_set = entity_set | find_entity_term(sentence)
      ent_by_sen[sentence] = entity_set                                   # sentence : entities
      buildListWithSet1(sen_by_paragraph, str(paragraph_idx), sentence)   # paragraph_idx : sentences

  print "\tBuilding entity-graph from input text...", datetime.datetime.now()
  import networkx as nx
  import operator

  input_data = []
  precedable_node = None
  pivot_node_collection = {}
  for target_p_idx, sentences in sorted(sen_by_paragraph.items()):
    ent_in_paragraph = set([])
    for sen in sentences:
      ent_in_paragraph = ent_in_paragraph | ent_by_sen[sen]
      for (a,b) in findsubset(ent_by_sen[sen], 2):
        clue =  sen.replace('[','').replace(']','').strip()
        input_data.append((clue.split(' '), a, b, 'unknown', sen))
        input_data.append((clue.split(' '), b, a, 'unknown', sen))

    # Continuously take over the central entity of the previous paragraph
    if precedable_node != None:
      ent_in_paragraph.add(precedable_node)

    G = nx.DiGraph()
    Edges = []
    for (a,b) in findsubset(ent_in_paragraph, 2):
      try:
        weight_ab = len(kb_memory[a+sep+b])
        Edges.append((a, b, weight_ab))
      except KeyError:
        try:
          weight_ba = len(kb_memory[b+sep+a])
          Edges.append((b, a, weight_ba))
        except KeyError:
          continue
    G.add_weighted_edges_from(Edges)
    stats = nx.out_degree_centrality(G)

    try:
      center_node = max(stats.iteritems(), key=operator.itemgetter(1))[0]
    except ValueError:
      continue

    precedable_node = center_node
    for sen in sentences:
      extended_sen = center_node + ' ' + sen
      if center_node not in ent_by_sen[sen]:
        for e in ent_by_sen[sen]:
          clue =  extended_sen.replace('[','').replace(']','').strip()
          input_data.append((clue.split(' '), center_node, e, 'unknown', sen))


  print "\tPredicting...", datetime.datetime.now()
  sen_data = []
  def read_data(clue, sbj, obj, relation):
    sen_data.append((clue, relation)) 

  tagged_input_data = [(read_data(row[0], row[1], row[2], row[3]), row[3]) for row in input_data]


  from collections import namedtuple
  WeaklyLabeledSentence = namedtuple('WeaklyLabeledSentence', 'words tags')
  tagged_test_docs1 = [WeaklyLabeledSentence(d, [c]) for d, c in sen_data]

  import pickle
  clf = pickle.load(open(classifier_model_filename)) 
  b2k_sent_model.random.seed(0)
  test_arrays1 = [b2k_sent_model.infer_vector(doc.words) for doc in tagged_test_docs1]
  prob1 = clf.predict_proba(test_arrays1)

  b2k_output_tmp = b2k_output + '_tmp'
  b2k_output_tmp2 = b2k_output + '_tmp2'
  b2k_output_raw = b2k_output + '_raw'

  fo = open(b2k_output_tmp, 'wb')
  fo2 = open(b2k_output_tmp2, 'wb')
  i=0

  for x in input_data:
    tmp_dict = {}

    given_sentence  = x[0]
    given_sbj       = x[1]
    target_obj      = x[2]
    ori_sentence    = x[4]

    given_sentence = ' '.join(given_sentence)

    for v1, v2 in zip (clf.classes_, prob1[i]):
      tmp_dict[v1] = v2

    for predict_relation, predict_prob_score  in sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True):
      fo.write('%s\t%s\t%s\t%s\t%s\n' % (ori_sentence, given_sbj, predict_relation, target_obj, predict_prob_score))
      fo2.write('%s\t%s\t%s\t%s:-:%s\n' % (given_sentence, given_sbj, predict_relation, target_obj, predict_prob_score))
    i += 1
  fo.close()
  fo2.close()

  print "\tReliability filtering...", datetime.datetime.now()
  fo_triple = open(b2k_output, 'wb')
  fo = open(b2k_output_raw, 'wb')
  min_score = 0
  triple_rlt = {}
  for row in open(b2k_output_tmp2):
    bunch, score = row.strip().split(sep)
    sentence, sbj, rel, obj = bunch.split('\t')

    try:
      if rel in p_sbj_info[sbj] and rel in p_obj_info[obj]:
        try:
          triple_rlt[sentence+sep+sbj+sep+obj]
        except KeyError:
          triple_rlt[sentence+sep+sbj+sep+obj] = rel
          if float(score) >= min_score:
            try:
              if rel in kb_memory[sbj+sep+obj]:
                score = '1.0'
            except KeyError:
              pass
            
            fo.write('%s\t%s\t%s\t.\t%s\t%s\n' % (sbj, rel, obj, score, sentence))
            fo_triple.write('%s\t%s\t%s\t.\t%s\n' % (sbj, rel, obj, score))

    except KeyError:
      continue
  fo.close()
  fo_triple.close()
  print '[END]', sys.argv[0], datetime.datetime.now()

  import os
  os.remove(b2k_output_tmp2)

@post('/b2k_process/')
def b2k_listen_process():
  inputs = request.params.get('inputs')
  b2k_input, b2k_output = inputs.split(',')
  process_b2k_extraction(b2k_input, b2k_output)
  return "DONE"

run(host='localhost', port=2311, debug=True)

