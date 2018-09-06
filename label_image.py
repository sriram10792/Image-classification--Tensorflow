#Function to do image classification

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
import urllib

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name, input_height, input_width,
				input_mean, input_std):
    
  input_name = "file_reader"
  print(file_name)
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result



def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def classification_function(path):
    model_file = "retrained_graph.pb"
    file_name=(path)
    print(file_name)
    label_file = "retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    """calls load_graph function"""
    graph = load_graph(model_file)
    """calls read from tensor function"""
    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)
    #print(t)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    results = np.squeeze(results)

    """calls load_labels function"""
    labels = load_labels(label_file)

    results_dictionary=dict(zip(labels,results))
    sorted_results_dictionary=sorted(results_dictionary.items(),key=lambda x: x[1],reverse=True)
    results_pandas=pd.DataFrame(sorted_results_dictionary)
    index_column=[1,2,3,4,5,6,7,8]
    results_pandas.insert(loc=0,column='Index',value=index_column)
    results_pandas.columns=['Index','Category','Confidence']
    
    #possible options for orient = index,split,records,columns,values,table
    results_json=results_pandas.to_json(orient='records')
    print (results_pandas)
    
    return results_json



