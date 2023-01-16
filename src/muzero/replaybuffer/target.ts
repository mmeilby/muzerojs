export class MuZeroTarget {
  constructor (
    // The value target is the discounted root value of the search tree N steps
    // into the future, plus the discounted sum of all rewards until then
    public readonly value: number,
    // The reward is the achieved score for this target state
    public readonly reward: number,
    // The policy represents the probability vector to most likely success
    // (Number of child visits for each action at this target state - found by MCTS)
    public readonly policy: number[]
  ) {}
}
