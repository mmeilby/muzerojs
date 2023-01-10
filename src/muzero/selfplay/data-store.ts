import { MCTSState } from './entities'

/**
 *
 * @hidden
 * @internal
 * @export
 * @interface DataGateway
 * @template Key
 * @template Value
 */
export interface DataGateway<State> {
  get: (key: State) => MCTSState<State> | undefined
  set: (key: State, value: MCTSState<State>) => this
}

/**
 *
 * @hidden
 * @internal
 * @template State
 * @template Action
 */
export interface Collection<State> {
  get: (key: string) => MCTSState<State> | undefined
  set: (key: string, value: MCTSState<State>) => this
}

/**
 *
 * @hidden
 * @internal
 * @template State
 * @template Action
 */
export class TranspositionTable<State> implements DataGateway<State> {
  constructor (private readonly data_: Collection<State>) {}

  get (key: State): MCTSState<State> | undefined {
    const stringKey = `${String(key)}`
    return this.data_.get(stringKey)
  }

  set (key: State, value: MCTSState<State>): this {
    const stringKey = `${String(key)}`
    this.data_.set(stringKey, value)
    return this
  }
}

/**
 *
 * @hidden
 * @internal
 * @template Key
 * @template Value
 */
export class HashTable<State> implements Collection<State> {
  private readonly buckets_: Array<Map<string, MCTSState<State>>> = []
  constructor (private readonly bucketCount_: number) {
    for (let i = 0; i < this.bucketCount_; i++) {
      this.buckets_.push(new Map())
    }
  }

  hashFunction_ (key: string): number {
    let hash = 0
    if (key.length === 0) return hash
    for (let i = 0; i < key.length; i++) {
      hash = (hash << 5) - hash
      hash = hash + key.charCodeAt(i)
      hash = hash & hash // Convert to 32bit integer
    }
    return Math.abs(hash)
  }

  getBucketIndex_ (key: string): number {
    return this.hashFunction_(key) % this.bucketCount_
  }

  getBucket_ (key: string): Map<string, MCTSState<State>> {
    return this.buckets_[this.getBucketIndex_(key)]
  }

  set (key: string, value: MCTSState<State>): this {
    this.getBucket_(key).set(key, value)
    return this
  }

  get (lookupKey: string): MCTSState<State> | undefined {
    return this.getBucket_(lookupKey).get(lookupKey)
  }
}
