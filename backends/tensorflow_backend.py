
###################################      
        
### TensorFlow Backend ############

###################################
    def buildTensorFlowModel(self,Network):
        self.features, self.labels, self.logits, self.modelKwargs = Network()



    def tfCompileModel(self,Optimizer,Loss,Metrics):
        self.Loss = Loss(self.labels,self.logits,self.modelKwargs)
        self.Optimizer = Optimizer(self.Loss,self.modelKwargs)
        self.Metrics = Metrics(self.labels,self.logits,self.modelKwargs)
        
    def tfTrainModel(self,Epochs,BatchSize,Verbose):
        
#        init_op = tf.global_variables_initializer()
##        with self.Session as sess:
#            # initialise the variables
#        self.Session.run(init_op)
#        total_batch = int(self.dataSlice.X_train.shape[0]/BatchSize)
#        for epoch in range(Epochs):
#            avg_cost = 0
#            for i in range(total_batch):
#                
#                batch_x, batch_y = self.dataSlice.getRandomBatch(BatchSize)
#                
#                _, c = self.Session.run([self.Optimizer, self.Loss], 
#                                feed_dict={self.x: batch_x, self.y: batch_y})
#                avg_cost += c / total_batch
#                print(i,c,batch_x.shape,total_batch)
#            test_acc = self.Session.run(self.Metrics, 
#                           feed_dict={self.x: self.dataSlice.X_test, self.y: self.dataSlice.y_test})
#            print("Epoch:", str(epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: ","{:.3f}".format(test_acc))
#    
#        print("\nTraining complete!")
#        print(self.Session.run(self.Metrics, feed_dict={self.x: self.dataSlice.X_test, self.y: self.dataSlice.y_test}))
#
#        self.Session.close()


        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

        # Run the initializer
            sess.run(init)
        
            for step in range(1, Epochs+1):
                batch_x, batch_y = self.dataSlice.getRandomBatch(BatchSize)
                # Run optimization op (backprop)
                print(batch_x.shape,batch_y.shape)
                sess.run(self.Optimizer, feed_dict={self.features: batch_x, self.labels: batch_y})
                
                if step % 10:
                    # Calculate batch loss and accuracy
                    metrics = [self.Loss] + [self.Metrics]
                    losser, acc = sess.run(metrics, feed_dict={self.features: batch_x, self.labels: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(losser) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))


