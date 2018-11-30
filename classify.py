import tensorflow as tf
import sys
import os
import cv2
import numpy as np
import timeit
from twilio.rest import Client

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.GFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

account_sid = "AC0b6567ca0ee1a1509f43d9a2031b2b16"
auth_token = "051afdccb7d835ad6d2acfec57341b96"
client = Client(account_sid, auth_token)

# Find these values at https://twilio.com/user/account
def sendSMS(body):
	message = client.api.account.messages.create(to="+14803749369",from_="+17062258408",body=body)


def classify(image_data):
    print('********* Session Start *********')

#    with tf.Session() as sess:
    start_time = timeit.default_timer()

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    print('Tensor', softmax_tensor)
    print( 'Took {} seconds to feed data to graph'.format(timeit.default_timer() - start_time))
    start_time = timeit.default_timer()

    # This takes 2-5 seconds as well
    predictions = sess.run(softmax_tensor, {'Mul:0': image_data})
    print( 'Took {} seconds to perform prediction'.format(timeit.default_timer() - start_time))

    start_time = timeit.default_timer()

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    print('Took {} seconds to sort the predictions'.format(timeit.default_timer() - start_time))

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

    print('********* Session Ended *********')

    if (top_k[0] == 0):
        return True;
    return False;

with tf.Session() as sess:

    camera = cv2.VideoCapture(0)
    while(1):
        # Capture frame from camera
        ret, frame = camera.read()
        frame = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_CUBIC)

        # adhere to TS graph input structure
        numpy_frame = np.asarray(frame)
        numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        numpy_final = np.expand_dims(numpy_frame, axis=0)

        if (classify(numpy_final)):
            sendSMS("IMPORTANT ALERT: Firearm detected")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
#if image_path:
#
#    # Read the image_data
#    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
#    print(type(image_data))
#
#    # Loads label file, strips off carriage return
#    label_lines = [line.rstrip() for line
#                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]
#
#    # Unpersists graph from file
#    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
#        _ = tf.import_graph_def(graph_def, name='')
#
#    with tf.Session() as sess:
#        # Feed the image_data as input to the graph and get first prediction
#        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#
#        predictions = sess.run(softmax_tensor, \
#                 {'DecodeJpeg/contents:0': image_data})
#
#        # Sort to show labels of first prediction in order of confidence
#        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#        for node_id in top_k:
#            human_string = label_lines[node_id]
#            score = predictions[0][node_id]
#            print('%s (score = %.5f)' % (human_string, score))

camera.release()
cv2.destroyAllWindows()
