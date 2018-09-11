import numpy as np
import random
import tensorflow as tf
import datetime
import LoadData
#Get Data
Sequence_Length, Char_Size, char2id, id2char, X, Y = LoadData.GetData()
  
#Define Training Parameters
Batch_Size = 512
Steps = 10
Eta = 10.0
Log_Interval = 10
Test_Interval = 10
Hidden_Nodes = 1024
Test_Start = 'I am thinking that'
Checkpoint_Directory = 'ckpt'

#Create a checkpoint directory
if tf.gfile.Exists(Checkpoint_Directory):
    tf.gfile.DeleteRecursively(Checkpoint_Directory)
tf.gfile.MakeDirs(Checkpoint_Directory)

print('training data size:', len(X))
print('approximate steps per epoch:', int(len(X)/Batch_Size))   

#Given a probability of each character, return a likely character, one-hot encoded
def Sample(Prediction):
    r = random.uniform(0,1)
    s = 0
    char_id = len(Prediction) - 1
    for i in range(len(Prediction)):
        s += Prediction[i]
        if s >= r:
            char_id = i
            break
        
    char_id = np.argmax(Prediction)
    Char_One_Hot = np.zeros(shape=[Char_Size])
    Char_One_Hot[char_id] = 1.0
    return Char_One_Hot

#Set up a tensorflow graph    
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0)

    #Define Placeholders for data and labels    
    data = tf.placeholder(tf.float32, [Batch_Size, Sequence_Length, Char_Size])
    labels = tf.placeholder(tf.float32, [Batch_Size, Char_Size])
    
    #Define variables for weights and biases
    #Input gate
    w_ii = tf.Variable(tf.truncated_normal([Char_Size, Hidden_Nodes], -0.1, 0.1))
    w_io = tf.Variable(tf.truncated_normal([Hidden_Nodes, Hidden_Nodes], -0.1, 0.1))
    b_i = tf.Variable(tf.zeros([1, Hidden_Nodes]))
    #Forget gate
    w_fi = tf.Variable(tf.truncated_normal([Char_Size, Hidden_Nodes], -0.1, 0.1))
    w_fo = tf.Variable(tf.truncated_normal([Hidden_Nodes, Hidden_Nodes], -0.1, 0.1))
    b_f = tf.Variable(tf.zeros([1, Hidden_Nodes]))
    #Output gate
    w_oi = tf.Variable(tf.truncated_normal([Char_Size, Hidden_Nodes], -0.1, 0.1))
    w_oo = tf.Variable(tf.truncated_normal([Hidden_Nodes, Hidden_Nodes], -0.1, 0.1))
    b_o = tf.Variable(tf.zeros([1, Hidden_Nodes]))
    #Memory cell
    w_ci = tf.Variable(tf.truncated_normal([Char_Size, Hidden_Nodes], -0.1, 0.1))
    w_co = tf.Variable(tf.truncated_normal([Hidden_Nodes, Hidden_Nodes], -0.1, 0.1))
    b_c = tf.Variable(tf.zeros([1, Hidden_Nodes]))
    
    #Define LSTM operations in the cell
    def LSTM(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
        forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
        output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
        memory_cell = tf.tanh(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)
        state = forget_gate * state + input_gate * memory_cell
        output = output_gate * tf.tanh(state)
        return output, state
    
    
    #Define output and state layers
    output = tf.zeros([Batch_Size, Hidden_Nodes])
    state = tf.zeros([Batch_Size, Hidden_Nodes])

    #Build Outputs and labels from batch data and batch labels
    for i in range(Sequence_Length):
        output, state = LSTM(data[:, i, :], output, state)
        if i == 0:
            Outputs = output
            Labels = data[:, i+1, :]
        elif i != Sequence_Length - 1:
            Outputs = tf.concat([Outputs, output], 0)
            Labels = tf.concat([Labels, data[:, i+1, :]], 0)
        else:
            Outputs = tf.concat([Outputs, output], 0)
            Labels = tf.concat([Labels, labels], 0) 

    #Final CLassification layer
    w = tf.Variable(tf.truncated_normal([Hidden_Nodes, Char_Size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([Char_Size]))
    #Get final probability distribution
    logits = tf.matmul(Outputs, w) + b
    #Define Loss 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Labels))
    #Define Optimizer
    optimizer = tf.train.GradientDescentOptimizer(Eta).minimize(loss, global_step=global_step)

    #Graph For Testing 
    Test_Data = tf.placeholder(tf.float32, shape=[1, Char_Size])
    Test_Output = tf.Variable(tf.zeros([1, Hidden_Nodes]))
    Test_State = tf.Variable(tf.zeros([1, Hidden_Nodes]))
    #Rest State
    Reset_Test_State = tf.group(Test_Output.assign(tf.zeros([1, Hidden_Nodes])),Test_State.assign(tf.zeros([1, Hidden_Nodes])))

    Test_Output, Test_State = LSTM(Test_Data, Test_Output, Test_State)
    Test_Prediction = tf.nn.softmax(tf.matmul(Test_Output, w) + b)

#Train the LSTM
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    pointer = 0
    saver = tf.train.Saver()

    file = open('Text.txt', 'w')
    #Start Training    
    for step in range(Steps):
        pointer = pointer % len(X)
        
        #Full batch_size can be covered
        if pointer + Batch_Size <= len(X):
            Batch_Data = X[pointer: pointer + Batch_Size]
            Batch_Labels = Y[pointer: pointer + Batch_Size]
            pointer += Batch_Size
        else:
            Req = Batch_Size - (len(X) - pointer)
            Batch_Data = np.concatenate((X[pointer: len(X)], X[0: Req]))
            Batch_Labels = np.concatenate((Y[pointer: len(X)], Y[0: Req]))
            pointer = Req

        #Run the network on batch data and batch labels and get the loss
        _, Training_Loss = sess.run([optimizer, loss], feed_dict={data: Batch_Data, labels: Batch_Labels})
        
        #If Log time        
        if step % Log_Interval == 0:
            print('training loss at step %d: %.2f (%s)' % (step, Training_Loss, datetime.datetime.now()))
            file.write('training loss at step %d: %.2f (%s)\n' % (step, Training_Loss, datetime.datetime.now()))
        #If Test time
        if step % Test_Interval == 0:
            #Predict the next data over 500 words
            Reset_Test_State.run()
            Test_Generated = Test_Start
            
            for i in range(len(Test_Start) - 1):
                #Build Encoding for test char
                Test_Char = np.zeros((1, Char_Size))
                #Encode Test Char
                Test_Char[0, char2id[Test_Start[i]]] = 1.0
                _ = sess.run(Test_Prediction, feed_dict={Test_Data: Test_Char})
            
            Test_Char = np.zeros((1, Char_Size))
            Test_Char[0, char2id[Test_Start[-1]]] = 1.0
            
            for i in range(500):
                Prediction = Test_Prediction.eval({Test_Data: Test_Char})[0]
                #Get Char with highest probability.
                Next_Char_One_Hot = Sample(Prediction)
                #Encode next char.
                Next_Char = id2char[np.argmax(Next_Char_One_Hot)]
                #Append next char to test generated.
                Test_Generated += Next_Char
                Test_Char = Next_Char_One_Hot.reshape((1, Char_Size))
            #Write text to File
            file.write('.' * 80+'\n') 
            file.write(Test_Generated)
            file.write('\n'+ '.' * 80 + '\n')
            #Save Info to checkpoint
            saver.save(sess, Checkpoint_Directory + '/model', global_step = step)
    file.close()

#Testing the model on a particular text.
Test_Start = 'I plan to make the world a better place by'
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    model = tf.train.latest_checkpoint(Checkpoint_Directory)
    saver = tf.train.Saver()
    saver.restore(sess, model)

    reset_test_state.run()
    test_generated = Test_Start

    for i in range(len(Test_Start) - 1):
        test_X = np.zeros((1, Char_Size))
        test_X[0, char2id[Test_Start[i]]] = 1.0
        _ = sess.run(test_prediction, feed_dict={test_data: test_X})

    test_X = np.zeros((1, Char_Size))
    test_X[0, char2id[Test_Start[-1]]] = 1.0

    for i in range(500):
        prediction = test_prediction.eval({test_data: test_X})[0]
        next_char_one_hot = sample(prediction)
        next_char = id2char[np.argmax(next_char_one_hot)]
        test_generated += next_char
        test_X = next_char_one_hot.reshape((1, Char_Size))

    print(test_generated)