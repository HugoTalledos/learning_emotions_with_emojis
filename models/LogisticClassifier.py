import tensorflow as tf

class LigisticClassifier:
    def __init__(self, seq_max_len, state_size, vocab_size, num_classes):
        self.seq_max_len = seq_max_len
        self.state_size = state_size #Define la capacidad del modelo
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def build_model(self):
        '''placeholder - Entradas de datos al modelo
        variables que tf utiliza para optimizar 
        shape= None - No importa cuantos ejemplos se pase
        '''
        #Datos de entrada  | x - observaciones
        self.x = tf.placeholder(shape=[None, self.seq_max_len], dtype=tf.int32)
        x_one_hot = tf.one_hot(self.x, self.vocab_size) #transforma variables categoricas a numericas
        x_one_hot = tf.cast(x_one_hot, tf.float32)

        #Datos de entrada | y - Clases
        self.y = tf.placeholder(shape=[None], dtype=tf.int32)
        self.y_one_hot = tf.one_hot(self.y, self.num_classes, dtype=tf.float32)
        
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        # y = mx + b | b=biases
        weights = {
            'layer_0': tf.Variable(tf.random_normal([self.seq_max_len*self.vocab_size,
            self.num_classes]))
        }
        biases = {
            'layer_0': tf.Variable(tf.random_normal([self.num_classes]))
        }

        #self.seq_max_len*self.vocab_size --> Datos de entrada al modelo
        x_input = tf.reshape(x_one_hot, [-1, self.seq_max_len*self.vocab_size])

        #salida --> Entrada * pesos
        output = tf.matmul(x_input, weights['layer_0']) + biases['layer_0']

        self.logits = tf.sigmoid(output)
        self.probs = tf.nn.softmax(self.logits, axis=1)

        self.corrct_predictions = tf.equal(tf.argmax(self.probs, axis=1), tf.argmax(self.y_one_hot, axis=1))
        self.precision = tf.reduce_mean(tf.cast(self.corrct_predictions, tf.float32()))

    def step_training(self, learning_rate=0.01):
        #Cross entropy - distancia entre distribuciones
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_one_hot)) 

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        return loss, optimizer