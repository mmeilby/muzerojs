/**
This file defines helper functions for computing and formatting varying parts of the loss computation/ output
representation of the MuZero or AlphaZero neural networks.
*/

import * as tf from '@tensorflow/tfjs'

/*
export function safeL2norm(x: tf.Tensor, epsilon=1e-5) {
  // Compute L2-Norm with an epsilon term for numerical stability (TODO Open github issue for this?) """
  const l2norm = tf.sqrt(tf.sum(x.mul(x).mul(tf.scalar(epsilon))));
  return l2norm.bufferSync().get(0);
}
*/
/**
 Scale gradients for reverse differentiation proportional to the given scale.
 Does not influence the magnitude/ scale of the output from a given tensor (just the gradient).
 */
/*
export function scaleGradient(tensor: tf.Tensor, scale: number): tf.Tensor {
  return tensor;
//  return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)
}
*/
/**
 Wrapper function to infer the correct loss function given the output representation.
 :param prediction: tf.Tensor  Output of a neural network.
 :param target: tf.Tensor  Target output for a neural network.
 :return: tf.losses Loss function between target and prediction
 */
/*
export function scalarLoss(prediction: tf.Tensor, target: tf.Tensor): tf.Tensor {
  if ((tf.prod(prediction.shape).arraySync() as number[])[0] === prediction.shape[0]) {
    // Implies (batch_size, 1) --> Regression
    return tf.losses.meanSquaredError(target, prediction);
  }
  // Default: Cross Entropy
  return tf.losses.softmaxCrossEntropy(target, prediction)
}
*/

/**
 Scalar transformation of rewards to stabilize variance and reduce scale.
 :references: https://arxiv.org/pdf/1805.11593.pdf
 */
function atariRewardTransform (x: number, varEps = 0.001): number {
  return Math.sign(x) * (Math.sqrt(Math.abs(x) + 1) - 1) + varEps * x
}

/**
 Inverse scalar transformation of atari_reward_transform function as used in the canonical MuZero paper.
 :references: https://arxiv.org/pdf/1805.11593.pdf
 */
function inverseAtariRewardTransform (x: number, varEps = 0.001): number {
  return Math.sign(x) * (Math.pow(((Math.sqrt(1 + 4 * varEps * (Math.abs(x) + 1 + varEps)) - 1) / (2 * varEps)), 2) - 1)
}

/**
 Recast distributional representation of floats back to floats. As the bins are symmetrically oriented around 0,
 this is simply done by taking a dot-product of the vector that represents the bins' integer range with the
 probability bins. After recasting of the floats, the floats are inverse-transformed for the scaling function.
 @param x tf.Tensor: 2D-array of floats in distributional representation: len(scalars) x (support_size * 2 + 1)
 @param supportSize: int Number of bins indicating integer range symmetric around zero.
 @return number[] Array of size len(scalars) x 1

 Reference :
 Appendix F => Network Architecture
 Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
 */
export function supportToScalar (x: tf.Tensor, supportSize: number): number[] {
  if (supportSize === 0) {
    // Simple regression (support in this case can be the mean of a Gaussian)
    return x.gather(0).arraySync() as number[]
  }
  const bins = tf.range(-supportSize, supportSize + 1)
  const y = x.dot(bins).arraySync() as number[]
  return y.map(v => inverseAtariRewardTransform(v))
}

/**
 Cast a scalar or array of scalars to a distributional representation symmetric around 0.
 For example, the float 3.4 given a support size of 5 will create 11 bins for integers [-5, ..., 5].
 Each bin is assigned a probability value of 0, bins 4 and 3 will receive probabilities .4 and .6 respectively.
 @param x number[]: 1D-array of floats to be cast to distributional bins.
 @param supportSize: int Number of bins indicating integer range symmetric around zero.
 @return tf.Tensor Array of size len(x) x (support_size * 2 + 1)
 */
export function scalarToSupport (x: number[], supportSize: number): tf.Tensor {
  if (supportSize === 0) {
    // Simple regression (support in this case can be the mean of a Gaussian)
    return tf.tensor([x])
  }
  // Clip float to fit within the support_size. Values exceeding this will be assigned to the closest bin.
  const transformed = tf.tensor(x.map(v => atariRewardTransform(v))).clipByValue(-supportSize, supportSize - 1e-8)
  const floored = transformed.floor() // Lower-bound support integer
  const proportion = transformed.sub(floored) // Proportion between adjacent integers
  const bins = tf.buffer([x.length, 2 * supportSize + 1])
  for (let i = 0; i < x.length; i++) {
    const j = (floored.arraySync() as number[])[i] + supportSize
    const prop = (proportion.arraySync() as number[])[i]
    bins.set(1 - prop, i, j)
    bins.set(prop, i, j + 1)
  }
  return bins.toTensor()
}
