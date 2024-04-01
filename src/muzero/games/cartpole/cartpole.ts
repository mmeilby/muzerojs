/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Implementation based on: http://incompleteideas.net/book/code/pole.c
 */

/**
 * Cart-pole data.old exchange class
 */
export class CartPoleDataSet {
  public constructor (
    public x: number,
    public xDot: number,
    public theta: number,
    public thetaDot: number,
    public reward: number
  ) {
  }
}
/**
 * Cart-pole system simulator.
 *
 * In the control-theory sense, there are four state variables in this system:
 *
 *   - x: The 1D location of the cart.
 *   - xDot: The velocity of the cart.
 *   - theta: The angle of the pole (in radians). A value of 0 corresponds to
 *     a vertical position.
 *   - thetaDot: The angular velocity of the pole.
 *
 * The system is controlled through a single action:
 *
 *   - leftward or rightward force.
 */
export class CartPole {
  private readonly gravity: number = 9.8
  private readonly massCart: number = 1.0
  private readonly massPole: number = 0.1
  private readonly lengthPole: number = 0.5
  private readonly forceMag: number = 10.0
  private readonly tau: number = 0.02 // Seconds between state updates.

  private readonly totalMass: number
  private readonly poleMoment: number

  // Threshold values, beyond which a simulation will be marked as failed.
  private readonly xThreshold: number = 2.4
  private readonly thetaThreshold: number = 12 / 360 * 2 * Math.PI

  /**
   * Constructor of CartPole.
   */
  public constructor () {
    // Constants that characterize the system.
    this.totalMass = this.massCart + this.massPole
    this.poleMoment = this.massPole * this.lengthPole
  }

  /**
   * Set the state of the cart-pole system randomly.
   */
  public static getRandomState (): CartPoleDataSet {
    // The control-theory state variables of the cart-pole system.
    // Cart position, meters.
    const x = (Math.random() - 0.5) * 0.1
    // Cart velocity.
    const xDot = Math.random() - 0.5
    // Pole angle, radians.
    const theta = (Math.random() - 0.5) * 0.1
    // Pole angle velocity.
    const thetaDot = Math.random() - 0.5
    return new CartPoleDataSet(x, xDot, theta, thetaDot, 0)
  }

  /**
   * Update the cart-pole system using an action.
   * @param {number} action Only the sign of `action` matters.
   *   A value > 0 leads to a rightward force of a fixed magnitude.
   *   A value <= 0 leads to a leftward force of the same fixed magnitude.
   */
  public update (dataset: CartPoleDataSet, action: number): CartPoleDataSet {
    const force = action > 0 ? this.forceMag : -this.forceMag

    const cosTheta = Math.cos(dataset.theta)
    const sinTheta = Math.sin(dataset.theta)

    const temp = (force + this.poleMoment * dataset.thetaDot * dataset.thetaDot * sinTheta) / this.totalMass
    const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) / (this.lengthPole * (4 / 3 - this.massPole * cosTheta * cosTheta / this.totalMass))
    const xAcc = temp - this.poleMoment * thetaAcc * cosTheta / this.totalMass

    // Update the four state variables, using Euler's method.
    const x = dataset.x + this.tau * dataset.xDot
    const xDot = dataset.xDot + this.tau * xAcc
    const theta = dataset.theta + this.tau * dataset.thetaDot
    const thetaDot = dataset.thetaDot + this.tau * thetaAcc

    return new CartPoleDataSet(x, xDot, theta, thetaDot, 1)
  }

  /**
   * Determine whether this simulation is done.
   *
   * A simulation is done when `x` (position of the cart) goes out of bound
   * or when `theta` (angle of the pole) goes out of bound.
   *
   * @returns {bool} Whether the simulation is done.
   */
  public isDone (dataset: CartPoleDataSet): boolean {
    return dataset.x < -this.xThreshold || dataset.x > this.xThreshold ||
      dataset.theta < -this.thetaThreshold || dataset.theta > this.thetaThreshold
  }

  public toString (dataset: CartPoleDataSet): string {
    const cart = `Cart: x=${dataset.x.toFixed(3)} m, v=${dataset.xDot.toFixed(3)} m/s | `
    const pole = `Pole: a=${dataset.theta.toFixed(3)} rad, v=${dataset.thetaDot.toFixed(3)} rad/s | `
    const reward = `Reward: ${dataset.reward}`
    return cart.concat(pole, reward)
  }
}
