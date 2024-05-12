import { describe, test } from '@jest/globals'
import * as tf from '@tensorflow/tfjs-node-gpu'

describe('Muzero Self Play Unit Test:', () => {
  test('Check stopGradient', () => {
    // Define placeholders
    const x = tf.variable(tf.ones([3, 2]))
    const y = tf.variable(tf.zeros([3, 4]))

    // Define variables
    const w1 = tf.variable(tf.ones([2, 3]))
    const w2 = tf.variable(tf.ones([3, 4]))

    const scaleGradiant = (tensor: tf.Tensor, scale: number): tf.Tensor => {
      return tf.tidy(() => {
        const tidyTensor = tf.variable(tensor, false)
        return tensor.mul(scale).add(tidyTensor.mul(1 - scale))
      })
    }
    // Define the computation graph
    const loss = (): tf.Scalar => {
      const hidden = x.matMul(w1)
      const output = hidden.matMul(w2)
      const e = output.sub(y)
      return scaleGradiant(e, 0.5).norm().asScalar()
    }
    const optimizer = tf.train.sgd(1)

    // Run the TensorFlow.js session
    console.log('*****before gradient descent*****')
    console.log('w1---\n', w1.arraySync(), '\n', 'w2---\n', w2.arraySync())

    //      x.assign(tf.randomNormal([3, 2]));
    //      y.assign(tf.randomNormal([3, 4]));

    optimizer.minimize(loss)

    console.log('*****after gradient descent*****')
    console.log('w1---\n', w1.arraySync(), '\n', 'w2---\n', w2.arraySync())
  })
})
